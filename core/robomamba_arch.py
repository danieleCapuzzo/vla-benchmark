"""
Bridge to robo_mamba/ (https://github.com/lmzpai/roboMamba):
CLIP ViT-L/14@224 -> 3-layer projector -> Mamba-2.8B.

Uses robo_mamba's `MambaLLM` (llm.py) since the checkpoint's `llm.mamba.*` keys are HF's own
naming, so this loads the same weights through the actively-maintained path.
"""

from __future__ import annotations

import contextlib
import glob
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from config.constants import ROBOMAMBA_IMAGE_SIZE, SUITE_TO_DATASET_DIR
from config.paths import ROBOMAMBA_SRC
from core.core_utils import resize_frame, to_upright

IMAGE_SIZE = ROBOMAMBA_IMAGE_SIZE  # CLIP ViT-L/14@224, fed from 256px renders
HIDDEN_DIM = 2560  # Mamba-2.8B hidden size
ACTION_DIM = 7
PROMPT_TEMPLATE = "<|user|> <image>\n{instruction}\n<|assistant|>"

# Actions predicted per trunk forward, executed open-loop by the evaluator's queue.
ACTION_CHUNK = 8
# The head reads f(t-ACTION_CHUNK) and f(t). A single frame cannot tell descending-to-grasp
# from ascending-while-carrying (near-identical images, opposite gripper commands); the pair
# resolves it. The gap equals the chunk because the evaluator only calls predict() once per
# chunk, so at eval the previous feature is always ACTION_CHUNK steps old.
HISTORY_FRAMES = 2

def to_model_input(image: np.ndarray, transform) -> torch.Tensor:
    """
    Raw LIBERO frame (any size) -> (3, IMAGE_SIZE, IMAGE_SIZE) float32.
    """
    temp = to_upright(image)
    temp = resize_frame(temp, IMAGE_SIZE)
    return transform(Image.fromarray(temp))


def _ensure_robomamba_importable() -> None:
    """robo_mamba/src has no __init__.py and its modules import as `model.vision`, so its
    src/ dir must be on sys.path."""
    if str(ROBOMAMBA_SRC) not in sys.path:
        sys.path.insert(0, str(ROBOMAMBA_SRC))


def build_trunk(checkpoint_path: Path, device: str, dtype: torch.dtype = torch.bfloat16):
    """
    Builds the frozen VLM trunk (vision + projector + Mamba) and loads `checkpoint_path`.
    Returns: 
        a `LinearVLM` in eval mode with all params frozen.
    """
    _ensure_robomamba_importable()
    from accelerate import init_empty_weights

    from model.llm import MambaLLM
    from model.vision import Vision
    from model.vlm import LinearVLM

    vision = Vision("CLIP224")
    with init_empty_weights():
        # removes naughty warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*copying from a non-meta parameter.*")
            llm = MambaLLM("mamba-2.8b")

    # create VLM and assert for errors
    trunk = LinearVLM(vision, llm)
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    key_counts = {"vision": 0, "projector": 0, "llm": 0}
    for key in state_dict:
        key_counts[key.split(".", 1)[0]] += 1
    assert key_counts == {"vision": 295, "projector": 6, "llm": 643}, (
        f"unexpected checkpoint key layout: {key_counts}"
    )

    missing, unexpected = trunk.load_state_dict(state_dict, strict=False, assign=True)
    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
    assert not missing, f"checkpoint left params unset: {missing}"

    leftover_meta = [n for n, p in trunk.named_parameters() if p.is_meta]
    assert not leftover_meta, f"params still on meta device after load: {leftover_meta}"

    trunk.tokenizer.pad_token = trunk.tokenizer.eos_token
    trunk = trunk.to(device=device, dtype=dtype).eval()
    trunk.requires_grad_(False)
    return trunk


@torch.no_grad()
def encode_batch(trunk, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Gets the frozen trunk output for a whole batch and mean pools the L language tokens.
    Recall the structure [256x vision  |  L text] tokens appended by the mamba llm.
    """
    inputs_embeds = trunk.encode(pixel_values, input_ids)
    # skip allocating mamba cache, less overhead
    hidden_states = trunk.llm.mamba.backbone(
        inputs_embeds=inputs_embeds, use_cache=False
    ).last_hidden_state

    num_vision_tokens = hidden_states.shape[1] - input_ids.shape[1]
    return hidden_states[:, num_vision_tokens:].mean(dim=1)


HEAD_WIDTH = 1024
HEAD_BLOCKS = 2


class _MLPResBlock(nn.Module):
    """Pre-norm residual block: x + Linear(GELU(Linear(LayerNorm(x))))."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RoboMambaHead(nn.Module):
    """
    LIBERO action head, MLP-ResNet:
    LayerNorm(5120) -> Linear(1024) -> HEAD_BLOCKS residual blocks -> LayerNorm -> chunk*7.

    Residual + pre-norm so gradients reach the input projection undiminished; the plain
    stack had to push all its signal through two GELUs in series.
    """

    def __init__(
        self, in_dim: int = HIDDEN_DIM * HISTORY_FRAMES, chunk: int = ACTION_CHUNK
    ) -> None:
        super().__init__()
        self.chunk = chunk
        self.stem = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, HEAD_WIDTH))
        self.blocks = nn.Sequential(*[_MLPResBlock(HEAD_WIDTH) for _ in range(HEAD_BLOCKS)])
        self.out = nn.Sequential(
            nn.LayerNorm(HEAD_WIDTH), nn.Linear(HEAD_WIDTH, chunk * ACTION_DIM)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, in_dim) -> (B, chunk, ACTION_DIM)."""
        x = self.blocks(self.stem(features))
        return self.out(x).unflatten(-1, (self.chunk, ACTION_DIM))


def save_checkpoint(path: Path, head: RoboMambaHead, meta: Dict[str, Any]) -> None:
    """
    Saves the head-only checkpoint (~15MB).
    Frozen trunk referenced via meta[trunk_checkpoint].
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(), "meta": meta}, path)


def load_head(path: Path, device: str) -> Tuple[RoboMambaHead, Dict[str, Any]]:
    """
    Loads the head-only checkpoint (~15MB).
    Frozen trunk referenced via meta[trunk_checkpoint].
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["head_state_dict"]
    # shapes come from the weights, not meta, so a checkpoint can never disagree with itself
    head = RoboMambaHead(
        in_dim=state_dict["stem.1.weight"].shape[1],
        chunk=state_dict["out.1.weight"].shape[0] // ACTION_DIM,
    )
    head.load_state_dict(state_dict)
    head = head.to(device).eval()
    return head, ckpt["meta"]


def build_history_chunks(
    features: torch.Tensor, actions: torch.Tensor, demo_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flat per-frame cache -> training pairs, one sample per frame.

    Returns (X (N, HIDDEN_DIM*HISTORY_FRAMES), Y (N, ACTION_CHUNK, ACTION_DIM), demo_ids (N,)).
    X is [f(t-ACTION_CHUNK), f(t)], Y the next ACTION_CHUNK actions. Both clamp at demo
    boundaries, matching what the wrapper does on its first call and at episode end.
    """
    xs, ys, ids = [], [], []
    for demo in torch.unique(demo_ids):
        # ascending indices == temporal order (each demo's frames are contiguous)
        idx = (demo_ids == demo).nonzero(as_tuple=True)[0]
        f, a, n = features[idx], actions[idx], len(idx)
        t = torch.arange(n)
        prev = (t - ACTION_CHUNK).clamp(min=0)
        chunk = (t[:, None] + torch.arange(ACTION_CHUNK)).clamp(max=n - 1)
        xs.append(torch.cat([f[prev], f], dim=1))
        ys.append(a[chunk])
        ids.append(demo_ids[idx])
    return torch.cat(xs), torch.cat(ys), torch.cat(ids)


# --------------------------------------------------------------------------------------
# LIBERO hdf5 loading + feature-cache extraction (stage 1 of train_robomamba.py)
# --------------------------------------------------------------------------------------

def _demo_files(data_dir: Path) -> List[Path]:
    files = sorted(Path(f) for f in glob.glob(str(data_dir / "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")
    return files


def feature_cache_path(output_dir: Path, suite: str) -> Path:
    """Cache path for a suite's extracted features."""
    return output_dir / "robomamba" / suite / "features.pt"


def _task_instructions(suite: str) -> Dict[str, str]:
    """Maps `{task.name}` -> task language from the LIBERO benchmark registry.

    Regenerated hdf5s carry no `problem_info` attr, so instructions come from the filename.
    """
    from libero.libero import benchmark

    # get_benchmark_dict() prints a banner that would corrupt the extraction tqdm bar.
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        task_suite = benchmark.get_benchmark_dict()[SUITE_TO_DATASET_DIR[suite]]()
    return {
        task_suite.get_task(i).name: task_suite.get_task(i).language
        for i in range(task_suite.n_tasks)
    }


def iter_suite_frames(data_dir: Path, suite: str):
    """
    Yields (instruction, images, actions, demo_ids) once per task hdf5 file.
    """
    instructions = _task_instructions(suite)
    demo_id = 0
    for path in _demo_files(data_dir):
        stem = path.stem.removesuffix("_demo")
        if stem not in instructions:
            raise KeyError(f"{path.name} has no matching task in the {suite} benchmark registry")
        instruction = instructions[stem]
        images: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        demo_ids: List[int] = []
        with h5py.File(path, "r") as f:
            for demo_key in f["data"]:
                demo = f["data"][demo_key]
                demo_images = demo["obs"]["agentview_rgb"][:]
                images.extend(demo_images)
                actions.extend(demo["actions"][:].astype(np.float32))
                demo_ids.extend([demo_id] * len(demo_images))
                demo_id += 1
        yield instruction, images, actions, demo_ids


def suite_frame_count(data_dir: Path) -> int:
    """Total frames across the suite, to preallocate the cache tensors."""
    total = 0
    for path in _demo_files(data_dir):
        with h5py.File(path, "r") as f:
            total += sum(f["data"][k]["actions"].shape[0] for k in f["data"])
    return total


def extract_features(trunk, data_dir: Path, suite: str, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One frozen-trunk pass over every frame in `data_dir`, one task file at a time.
    `suite` resolves task instructions via the LIBERO benchmark registry (see
    `_task_instructions`).

    Returns (features fp16 (N, 2560), actions fp32 (N, 7), demo_ids int64 (N,)) on CPU.
    """
    n_frames = suite_frame_count(data_dir)
    features_out = torch.empty(n_frames, HIDDEN_DIM, dtype=torch.float16)
    actions_out = torch.empty(n_frames, ACTION_DIM, dtype=torch.float32)
    demo_ids_out = torch.empty(n_frames, dtype=torch.int64)

    write_ptr = 0
    pbar = tqdm(total=n_frames, desc="extracting features", unit="frame")
    for instruction, images, actions, demo_ids in iter_suite_frames(data_dir, suite):
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        input_ids_row = trunk.tokenizer(prompt, return_tensors="pt").input_ids

        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch_actions = actions[start : start + batch_size]
            batch_demo_ids = demo_ids[start : start + batch_size]

            pixel_values = torch.stack(
                [to_model_input(img, trunk.transform) for img in batch_images]
            ).to(device=device, dtype=trunk.llm.mamba.dtype)
            input_ids = input_ids_row.repeat(len(batch_images), 1).to(device)

            features = encode_batch(trunk, pixel_values, input_ids)

            n = len(batch_images)
            features_out[write_ptr : write_ptr + n] = features.to("cpu", dtype=torch.float16)
            actions_out[write_ptr : write_ptr + n] = torch.from_numpy(np.stack(batch_actions))
            demo_ids_out[write_ptr : write_ptr + n] = torch.tensor(batch_demo_ids, dtype=torch.int64)
            write_ptr += n
            pbar.update(n)
    pbar.close()

    assert write_ptr == n_frames
    return features_out, actions_out, demo_ids_out
