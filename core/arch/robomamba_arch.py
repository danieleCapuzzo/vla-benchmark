"""
Bridge to robo_mamba/ (https://github.com/lmzpai/roboMamba):
CLIP ViT-L/14@224 -> 3-layer projector -> Mamba-2.8B.

Uses robo_mamba's `MambaLLM` (llm.py) since the checkpoint's `llm.mamba.*` keys are HF's own
naming, so this loads the same weights through the actively-maintained path.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from config.constants import ROBOMAMBA_IMAGE_SIZE
from config.paths import ROBOMAMBA_SRC
from core.core_utils import resize_frame, to_upright

IMAGE_SIZE = ROBOMAMBA_IMAGE_SIZE  # CLIP ViT-L/14@224, fed from 256px renders
HIDDEN_DIM = 2560  # Mamba-2.8B hidden size
ACTION_DIM = 7
PROMPT_TEMPLATE = "<|user|> <image>\n{instruction}\n<|assistant|>"

# Actions predicted per trunk forward, executed open-loop by the evaluator's queue.
ACTION_CHUNK = 8

# head parameters
HEAD_WIDTH = 1024
HEAD_BLOCKS = 2

def _ensure_robomamba_importable() -> None:
    """robo_mamba/src has no __init__.py and its modules import as `model.vision`, so its
    src/ dir must be on sys.path."""
    if str(ROBOMAMBA_SRC) not in sys.path:
        sys.path.insert(0, str(ROBOMAMBA_SRC))

def to_model_input(image: np.ndarray, transform) -> torch.Tensor:
    """
    Raw LIBERO frame (any size) -> (3, IMAGE_SIZE, IMAGE_SIZE) float32.
    """
    temp = to_upright(image)
    temp = resize_frame(temp, IMAGE_SIZE)
    return transform(Image.fromarray(temp))


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


def encode_features(
    trunk,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Gradient-transparent trunk forward (base or LoRA-adapted) with mask-weighted mean pooling
    over the L language tokens. Recall the structure [256x vision | L text] tokens appended by
    the mamba llm.
    Args:
        trunk: LinearVLM, base or PEFT-wrapped (attribute access delegates through to it).
        pixel_values: (B, 3, IMAGE_SIZE, IMAGE_SIZE).
        input_ids: (B, L) right-padded token ids (Mamba is causal-recurrent, so trailing pads
            can't leak into earlier hidden states -- left padding would).
        attention_mask: (B, L), 1 for real tokens / 0 for right-padding. None means every row is
            unpadded (single instruction repeated across the batch, e.g. Stage 1 extraction).
    Returns:
        (B, HIDDEN_DIM) pooled language-token features.
    """
    inputs_embeds = trunk.encode(pixel_values, input_ids)
    # skip allocating mamba cache, less overhead
    hidden_states = trunk.llm.mamba.backbone(
        inputs_embeds=inputs_embeds, use_cache=False
    ).last_hidden_state

    num_vision_tokens = hidden_states.shape[1] - input_ids.shape[1]
    lang_hidden = hidden_states[:, num_vision_tokens:]
    if attention_mask is None:
        return lang_hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(lang_hidden.dtype)
    return (lang_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def encode_batch(
    trunk,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """No-grad wrapper around `encode_features`, for frozen-trunk feature caching and eval."""
    return encode_features(trunk, pixel_values, input_ids, attention_mask)


def force_mamba_module_path(trunk) -> int:
    """
    Flips every MambaMixer submodule's `.training` to False so HF's fused fast-path forward
    (active whenever `mamba_ssm`/`causal_conv1d` are installed) takes its module-call branch
    (`self.x_proj(x)`) rather than its training branch, which reads raw `.weight` tensors and so
    silently bypasses LoRA on x_proj/out_proj (PEFT's lora.Linear only intercepts calls, not
    `.weight` access -- `dt_proj` uses a raw weight on both branches and is excluded from LoRA
    entirely for the same reason).

    Safe to call under `.train()`: MambaMixer has no other training-dependent behaviour, and
    `MambaModel.training` is left untouched, so gradient checkpointing and `use_cache=False`
    keep working. Must be re-applied after every `.train()` call, since it recurses and would
    otherwise flip mixers back.
    Args:
        trunk: LinearVLM (base or PEFT-wrapped).
    Returns:
        Number of MambaMixer modules patched, for a startup sanity check.
    """
    from transformers.models.mamba.modeling_mamba import MambaMixer

    n = 0
    for module in trunk.modules():
        if isinstance(module, MambaMixer):
            module.training = False
            n += 1
    return n


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
    LayerNorm(2560) -> Linear(1024) -> HEAD_BLOCKS residual blocks -> LayerNorm -> chunk*7.
    """

    def __init__(self, in_dim: int = HIDDEN_DIM, chunk: int = ACTION_CHUNK) -> None:
        super().__init__()
        self.chunk = chunk
        self.stem = nn.Sequential(
            nn.LayerNorm(in_dim), 
            nn.Linear(in_dim, HEAD_WIDTH)
        )
        self.blocks = nn.Sequential(
            *[_MLPResBlock(HEAD_WIDTH) for _ in range(HEAD_BLOCKS)]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(HEAD_WIDTH), nn.Linear(HEAD_WIDTH, chunk * ACTION_DIM)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, in_dim) -> (B, chunk, ACTION_DIM)."""
        x = self.blocks(self.stem(features))
        return self.out(x).unflatten(-1, (self.chunk, ACTION_DIM))


def save_checkpoint(path: Path, head: RoboMambaHead, meta: Dict[str, Any]) -> None:
    """
    Saves the entire checkpoint: head weights (~15MB) plus meta.
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

    # shapes come from the weights
    head = RoboMambaHead(
        in_dim=state_dict["stem.1.weight"].shape[1],
        chunk=state_dict["out.1.weight"].shape[0] // ACTION_DIM,
    )
    head.load_state_dict(state_dict)
    head = head.to(device).eval()
    return head, ckpt["meta"]


def load_checkpoint(
    head_path: Path | None,
    device: str,
    trunk_path: Path | None = None,
) -> Tuple[RoboMambaHead | None, Any, Dict[str, Any]]:
    """
    Loads the entire checkpoint: head (from `head_path`) and the frozen trunk it
    references (or `trunk_path`, if given as an override).
    Args:
        head_path: Head checkpoint path, as written by `save_checkpoint`. None means
            "trunk only" -- the returned head is None.
        device: Device to load the trunk and head onto.
        trunk_path: Overrides the trunk checkpoint referenced by the head's own meta,
            enabling separate trunk/head files. Required if `head_path` is None (there is
            no meta to resolve a default trunk path from).

    Returns:
        A (head, trunk, meta) tuple. `head` is None for a trunk-only load. `meta` is the
        head checkpoint's metadata, or `{}` for a trunk-only load.

    Raises:
        ValueError: If both `head_path` and `trunk_path` are None.
    """
    if head_path is None:
        if trunk_path is None:
            raise ValueError("load_checkpoint() needs at least one of head_path/trunk_path.")
        head, meta = None, {}
    else:
        head, meta = load_head(head_path, device=device)

    resolved_trunk_path = trunk_path if trunk_path is not None else meta.get("trunk_checkpoint")
    if resolved_trunk_path is None:
        raise ValueError(
            f"Checkpoint '{head_path}' has no trunk_checkpoint in its meta; pass trunk_path."
        )
    trunk = build_trunk(Path(resolved_trunk_path), device=device)

    return head, trunk, meta
