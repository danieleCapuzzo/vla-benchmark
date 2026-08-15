"""
PyTorch Dataset over regenerated LIBERO HDF5 files (datasets/<suite>_regen/). S
erves both Stage 1 frozen-trunk feature extraction (augment=False) and online
LoRA training (augment=True).
"""

from __future__ import annotations

import contextlib
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.constants import SUITE_TO_DATASET_DIR
from core.arch.robomamba_arch import (
    ACTION_CHUNK,
    ACTION_DIM,
    HIDDEN_DIM,
    IMAGE_SIZE,
    PROMPT_TEMPLATE,
    encode_batch,
)
from core.core_utils import normalize_actions, resize_frame, to_upright

# CLIP ViT-L/14 (openai) normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# image augmentation config
# matched to OpenVLA-OFT
_CROP_SCALE = 0.9
_BRIGHTNESS_DELTA = 0.2
_CONTRAST_RANGE = (0.8, 1.2)
_SATURATION_RANGE = (0.8, 1.2)
_HUE_DELTA = 0.05


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """converts RGB image to HSV"""
    r, g, b = img.unbind(-1)
    maxc, _ = img.max(-1)
    minc, _ = img.min(-1)
    diff = (maxc - minc).clamp_min(1e-8)
    s = torch.where(maxc > 0, (maxc - minc) / maxc.clamp_min(1e-8), torch.zeros_like(maxc))
    rc, gc, bc = (maxc - r) / diff, (maxc - g) / diff, (maxc - b) / diff
    h = torch.where(r == maxc, bc - gc, torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.where(maxc == minc, torch.zeros_like(h), (h / 6.0) % 1.0)
    return torch.stack([h, s, maxc], dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """converts HSV image to RGB"""
    h, s, v = hsv.unbind(-1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.long() % 6
    cases = torch.stack(
        [
            torch.stack([v, t, p], -1),
            torch.stack([q, v, p], -1),
            torch.stack([p, v, t], -1),
            torch.stack([p, q, v], -1),
            torch.stack([t, p, v], -1),
            torch.stack([v, p, q], -1),
        ],
        dim=0,
    )
    return torch.gather(cases, 0, i[None, ..., None].expand(1, *i.shape, 3))[0]


def _random_resized_crop(img: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Torch reimplementation of dlimp `random_resized_crop`, correctness numerically verified
    """
    H, W, _ = img.shape
    side = _CROP_SCALE**0.5  # new_height == new_width since ratio == 1
    u = torch.rand((), generator=generator).item()
    off = u * (1.0 - side)
    y_frac = torch.linspace(off, off + side, H)
    x_frac = torch.linspace(off, off + side, W)
    grid_y, grid_x = torch.meshgrid(y_frac * 2 - 1, x_frac * 2 - 1, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    chw = img.permute(2, 0, 1).unsqueeze(0)
    out = F.grid_sample(chw, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
    return out.squeeze(0).permute(1, 2, 0)


def _augment(img: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """
    Applies data augmentation to images in a pre-determined order:
    crop, brightness, contrast, saturation, hue, clipping to [0,1] after every op.
    """
    img = _random_resized_crop(img, generator).clamp(0, 1)

    delta = (torch.rand((), generator=generator).item() * 2 - 1) * _BRIGHTNESS_DELTA
    img = (img + delta).clamp(0, 1)

    lo, hi = _CONTRAST_RANGE
    factor = lo + torch.rand((), generator=generator).item() * (hi - lo)
    mean = img.mean(dim=(0, 1), keepdim=True)  # per-channel, not grayscale
    img = ((img - mean) * factor + mean).clamp(0, 1)

    lo, hi = _SATURATION_RANGE
    factor = lo + torch.rand((), generator=generator).item() * (hi - lo)
    hsv = _rgb_to_hsv(img)
    hsv[..., 1] = (hsv[..., 1] * factor).clamp(0, 1)
    img = _hsv_to_rgb(hsv).clamp(0, 1)

    delta = (torch.rand((), generator=generator).item() * 2 - 1) * _HUE_DELTA
    hsv = _rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + delta) % 1.0
    img = _hsv_to_rgb(hsv).clamp(0, 1)

    return img


def _chunk_actions(demo_actions: np.ndarray, frame_idx: int, chunk: int) -> np.ndarray:
    """
    Repeats the demo's final action once it runs past the demo's end.
    """
    n = demo_actions.shape[0]
    idx = np.arange(frame_idx, frame_idx + chunk).clip(max=n - 1)
    return demo_actions[idx]


def _demo_files(data_dir: Path) -> List[Path]:
    """Sorted list of `*.hdf5` files in `data_dir`."""
    files = sorted(Path(f) for f in glob.glob(str(data_dir / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def _task_instructions(suite: str) -> Dict[str, str]:
    """
    Maps `{task.name}` -> task language from the LIBERO benchmark registry.
    """
    from libero.libero import benchmark

    # get_benchmark_dict() prints a banner that would corrupt a tqdm bar.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        task_suite = benchmark.get_benchmark_dict()[SUITE_TO_DATASET_DIR[suite]]()
    return {
        task_suite.get_task(i).name: task_suite.get_task(i).language
        for i in range(task_suite.n_tasks)
    }


class LiberoHDF5Dataset(Dataset):
    """
    Frame-level dataset over a regenerated LIBERO suite's HDF5 files.
    Args:
        data_dir: Folder of `*.hdf5` demo files (e.g. `datasets/libero_spatial_regen`).
        suite: Suite name from `config.constants.SUITE_TO_DATASET_DIR` (e.g. "spatial"), used to
            resolve each task's language instruction from the LIBERO benchmark registry.
        augment: False applies only resize + CLIP normalize (byte-identical to `to_model_input`,
            used for Stage 1 caching and eval). True additionally applies OFT's exact image
            augmentation pipeline before normalization (online LoRA training).
        image_size: Output frame side length.
        action_chunk: Actions per sample; chunks are clamped (repeat the final action) at demo
            end, matching the eval wrapper's own open-loop execution.
        action_q01, action_q99: Optional per-dim (ACTION_DIM,) bounds. When given, actions are
            bounds-normalized to [-1, 1] (see `core.core_utils.normalize_actions`); None returns
            raw actions -- chunk[:, 0] is then exactly the frame's own raw action, which Stage 2
            uses to compute train-partition-only bounds after caching.
        seed: Base seed for the per-sample augmentation RNG (see `set_epoch`).
    """

    def __init__(
        self,
        data_dir: Path,
        suite: str,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
        action_chunk: int = ACTION_CHUNK,
        action_q01: Optional[np.ndarray] = None,
        action_q99: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.suite = suite
        self.augment = augment
        self.image_size = image_size
        self.action_chunk = action_chunk
        self.action_q01 = action_q01
        self.action_q99 = action_q99
        self.seed = seed
        self.epoch = 0

        self._files = _demo_files(self.data_dir)
        self._instructions = _task_instructions(suite)
        self._mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self._std = torch.tensor(CLIP_STD).view(3, 1, 1)

        # (file_idx, demo_key, frame_idx, demo_id) per frame; actions loaded eagerly per demo
        # (tiny), images stay lazy in HDF5 (file handles opened per-worker, see `_file`).
        self._index: List[Tuple[int, str, int, int]] = []
        self._demo_actions: List[np.ndarray] = []
        demo_id = 0
        for file_idx, path in enumerate(self._files):
            stem = path.stem.removesuffix("_demo")
            if stem not in self._instructions:
                raise KeyError(f"{path.name} has no matching task in the {suite} benchmark registry")
            with h5py.File(path, "r") as f:
                for demo_key in f["data"]:
                    actions = f["data"][demo_key]["actions"][:].astype(np.float32)
                    self._demo_actions.append(actions)
                    for frame_idx in range(actions.shape[0]):
                        self._index.append((file_idx, demo_key, frame_idx, demo_id))
                    demo_id += 1

        self._handles: Dict[int, h5py.File] = {}

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch used to seed per-sample augmentation RNG. Call before each epoch's
        iteration -- with `persistent_workers=False` (the default), workers are re-forked from
        this state, so augmentation is reproducible across resume and independent of worker
        count/order (mirrors dlimp's own stateless-seed design)."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self._index)

    def _file(self, file_idx: int) -> h5py.File:
        # h5py.File handles aren't fork-safe -- opened lazily in the worker process that needs them.
        if file_idx not in self._handles:
            self._handles[file_idx] = h5py.File(self._files[file_idx], "r")
        return self._handles[file_idx]

    def _seed_for(self, index: int) -> int:
        return (self.seed * 1_000_003 + self.epoch * 97 + index) & 0xFFFFFFFF

    def __getitem__(self, index: int) -> Dict[str, object]:
        file_idx, demo_key, frame_idx, demo_id = self._index[index]
        f = self._file(file_idx)
        image = f["data"][demo_key]["obs"]["agentview_rgb"][frame_idx]
        instruction = self._instructions[self._files[file_idx].stem.removesuffix("_demo")]

        frame = to_upright(image)
        frame = resize_frame(frame, self.image_size)
        img = torch.from_numpy(frame).float().div(255.0)  # (H,W,3) in [0,1]

        if self.augment:
            generator = torch.Generator().manual_seed(self._seed_for(index))
            img = _augment(img, generator)

        pixel_values = img.permute(2, 0, 1)  # (3,H,W)
        pixel_values = pixel_values.sub(self._mean).div(self._std)

        actions = _chunk_actions(self._demo_actions[demo_id], frame_idx, self.action_chunk)
        if self.action_q01 is not None:
            actions = normalize_actions(actions, self.action_q01, self.action_q99)

        return {
            "pixel_values": pixel_values,
            "actions": torch.from_numpy(actions.astype(np.float32)),
            "instruction": instruction,
            "demo_id": demo_id,
        }


def compute_action_bounds(data_dir: Path, suite: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-dim q01/q99 raw-action bounds over every frame in `data_dir`, with no train/val split. 
    Reads action arrays only (no images).
    Returns:
        (q01, q99), each (ACTION_DIM,) float32.
    """
    dataset = LiberoHDF5Dataset(data_dir, suite, augment=False)
    actions = np.concatenate(dataset._demo_actions, axis=0)
    q01 = np.quantile(actions, 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(actions, 0.99, axis=0).astype(np.float32)
    return q01, q99


def make_collate_fn(tokenizer):
    """
    Builds a DataLoader `collate_fn` that formats and tokenizes each sample's instruction with
    `PROMPT_TEMPLATE`, right-padding to the batch's longest prompt.
    Args:
        tokenizer: The trunk's tokenizer (`trunk.tokenizer`); mutated in place to pad right.
    Returns:
        A callable `(list[dict]) -> dict` with keys pixel_values, actions, demo_id, input_ids,
        attention_mask.
    """
    tokenizer.padding_side = "right"

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [PROMPT_TEMPLATE.format(instruction=b["instruction"]) for b in batch]
        tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "actions": torch.stack([b["actions"] for b in batch]),
            "demo_id": torch.tensor([b["demo_id"] for b in batch], dtype=torch.long),
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
        }

    return collate


def extract_features(
    trunk,
    data_dir: Path,
    suite: str,
    batch_size: int,
    device: str,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One frozen-trunk pass over every frame in `data_dir`, via `LiberoHDF5Dataset(augment=False)`.
    Returns:
        (features fp16 (N, HIDDEN_DIM), actions fp32 (N, ACTION_CHUNK, ACTION_DIM) -- raw, not
        bounds-normalized; chunk[:, 0] is each frame's own raw action -- demo_ids int64 (N,)) on
        CPU, in dataset order.
    """
    dataset = LiberoHDF5Dataset(data_dir, suite, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=make_collate_fn(trunk.tokenizer),
    )
    n = len(dataset)
    features_out = torch.empty(n, HIDDEN_DIM, dtype=torch.float16)
    actions_out = torch.empty(n, ACTION_CHUNK, ACTION_DIM, dtype=torch.float32)
    demo_ids_out = torch.empty(n, dtype=torch.int64)

    write_ptr = 0
    pbar = tqdm(total=n, desc="extracting features", unit="frame")
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device=device, dtype=trunk.llm.mamba.dtype)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        features = encode_batch(trunk, pixel_values, input_ids, attention_mask)

        b = pixel_values.shape[0]
        features_out[write_ptr : write_ptr + b] = features.to("cpu", dtype=torch.float16)
        actions_out[write_ptr : write_ptr + b] = batch["actions"]
        demo_ids_out[write_ptr : write_ptr + b] = batch["demo_id"]
        write_ptr += b
        pbar.update(b)
    pbar.close()

    assert write_ptr == n
    return features_out, actions_out, demo_ids_out
