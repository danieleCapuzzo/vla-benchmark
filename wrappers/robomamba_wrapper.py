"""
Adapter for RoboMamba (see https://github.com/lmzpai/roboMamba): frozen CLIP+Mamba trunk
fine-tuned per-suite with a small LIBERO action head (see core/robomamba_arch.py).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from config.constants import NUM_CAMERAS_BY_MODEL
from core.core_utils import denormalize_actions, env_gripper_to_rlds
from core.factory import PolicyFactory
from core.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper
from core.robomamba_arch import (
    HIDDEN_DIM,
    PROMPT_TEMPLATE,
    build_trunk,
    encode_batch,
    load_head,
    to_model_input,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_CAMERAS = NUM_CAMERAS_BY_MODEL["robomamba"]


def _resolve(path: str | Path) -> Path:
    """Repo-relative paths (as written in config/checkpoint meta) -> absolute."""
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


@PolicyFactory.register("robomamba")
class RoboMambaWrapper(PolicyWrapper):
    """Adapter pattern: adapts RoboMamba's frozen-trunk + head to the PolicyWrapper contract."""

    def __init__(self, checkpoint: str, unnorm_key: str) -> None:
        """
        Args:
            checkpoint: Path to a robomamba head checkpoint (.pt) for the target suite.
            unnorm_key: Unused but kept for constructor signature parity.
        """
        self._checkpoint = checkpoint
        self._trunk = None
        self._head = None
        self._meta: Dict[str, Any] = {}
        self._quant = "none"
        # per-dim action bounds (q01/q99); None means the head predicts raw actions
        self._action_q01: np.ndarray | None = None
        self._action_q99: np.ndarray | None = None
        # f(t-K) for the history pair; None on the first call of an episode
        self._prev_features: torch.Tensor | None = None

    def load(self, quant: str = "none", **kwargs: Any) -> None:
        """
        Loads a RoboMamba head checkpoint and the frozen trunk it was trained against.

        Args:
            quant: Must be "none"; RoboMamba has no quantized loading path.
            **kwargs: Unused.

        Raises:
            ValueError: if `quant != "none"`.
        """
        if quant != "none":
            raise ValueError(f"RoboMamba has no quantized loading path; got quant='{quant}'.")
        self._quant = quant

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.reset_peak_memory_stats()

        self._head, self._meta = load_head(_resolve(self._checkpoint), device=device)

        # absent for checkpoints trained before action normalization was added; predict()
        # then skips denormalization
        if self._meta.get("action_q01") is not None:
            self._action_q01 = np.asarray(self._meta["action_q01"], dtype=np.float32)
            self._action_q99 = np.asarray(self._meta["action_q99"], dtype=np.float32)

        self._trunk = build_trunk(_resolve(self._meta["trunk_checkpoint"]), device=device)

    def reset(self) -> None:
        """
        Drops the previous episode's feature so its history never leaks into the next.
        """
        self._prev_features = None

    def predict(self, observation: Dict[str, Any], task_description: str) -> PolicyOutput:
        """
        Runs one RoboMamba step: frozen trunk forward (language-pooled readout) -> head.
        Args:
            observation: Dict with "raw_obs" (the raw LIBERO env observation dict).
            task_description: Natural-language task instruction, always used.
        Returns:
            PolicyOutput with actions of shape (ACTION_CHUNK, 7), is_chunked=True.
        """
        device = next(self._trunk.parameters()).device
        pixel_values = to_model_input(
            observation["raw_obs"]["agentview_image"], self._trunk.transform
        ).unsqueeze(0).to(device=device, dtype=self._trunk.llm.mamba.dtype)

        prompt = PROMPT_TEMPLATE.format(instruction=task_description)
        input_ids = self._trunk.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # latency covers the model call only\
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        features = encode_batch(self._trunk, pixel_values, input_ids)

        # first call of episode has no history: 
        # pair f(t) with itself
        prev = features if self._prev_features is None else self._prev_features
        actions = self._head(torch.cat([prev, features], dim=-1).float())
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - t0

        self._prev_features = features
        actions = actions.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # order matters: the gripper remap
        #  assumes raw LIBERO units, so denormalize first
        if self._action_q01 is not None:
            actions = denormalize_actions(actions, self._action_q01, self._action_q99)
        actions = env_gripper_to_rlds(actions)

        return PolicyOutput(
            actions=actions.astype(np.float32),
            is_chunked=True,
            raw_latency_s=latency_s,
        )

    def get_metadata(self) -> PolicyMetadata:
        """
        Returns fully-populated metadata for the currently loaded checkpoint.
        """
        if self._trunk is None or self._head is None:
            raise RuntimeError("get_metadata() called before load()")
        num_params = sum(p.numel() for p in self._trunk.parameters()) + sum(
            p.numel() for p in self._head.parameters()
        )
        return PolicyMetadata(
            model_id="robomamba",
            variant=f"clip224_mamba2.8b_{self._meta.get('suite', '?')}",
            num_params=num_params,
            quant=self._quant,
            num_cameras=NUM_CAMERAS,
            action_chunk_size=self._head.chunk,
            attn_impl="n/a",  # Mamba SSM, no attention
            extra={
                "train_epoch": self._meta.get("epoch"),
                "trunk_checkpoint": self._meta.get("trunk_checkpoint"),
                "hidden_dim": HIDDEN_DIM,
            },
        )

    def get_vram_footprint_gb(self) -> float:
        """
        Returns peak allocated VRAM (GB) observed since the last reset_peak_memory_stats().
        """
        return torch.cuda.max_memory_allocated() / 1e9
