"""
Adapter for base OpenVLA (see https://github.com/openvla/openvla).

Wraps the official inference pipeline (single agentview camera, no action chunking) using
`wrappers/wrapper_utils.py` helpers -- Blackwell-safe (sdpa, never flash_attn), never
duplicates the official harness's paper-critical resize/crop math.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import AutoProcessor

from wrappers.wrapper_utils import (
    OPENVLA_INPUT_RESOLUTION,
    load_openvla_model,
    prepare_openvla_inputs,
    register_openvla_auto_classes,
)

# patch issue with the official harness assuming cwd == openvla/ (norm_stats lookup, etc.)
_OPENVLA_REPO_ROOT = Path(__file__).resolve().parent.parent / "openvla"
if str(_OPENVLA_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENVLA_REPO_ROOT))

from config.loader import resolve_unnorm_key  # noqa: E402
from core.factory import PolicyFactory  # noqa: E402
from core.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper  # noqa: E402


@PolicyFactory.register("openvla")
class OpenVlaWrapper(PolicyWrapper):
    """
    Adapter pattern: adapts base OpenVLA's `predict_action` API to the PolicyWrapper
    contract. Single third-person (agentview) camera only -- base OpenVLA has no wrist-image
    input path, unlike OFT.
    """

    def __init__(self, checkpoint: str, unnorm_key: str) -> None:
        """
        Creates the wrapper. Model weights are loaded lazily in `load()`.

        Args:
            checkpoint: HF Hub id (or local path) of the OpenVLA checkpoint for the target suite.
            unnorm_key: action un-normalization key matching this checkpoint (e.g. "libero_spatial").
        """
        self._checkpoint = checkpoint
        self._unnorm_key = unnorm_key
        self._vla = None
        self._processor = None
        self._quant = "none"
        self._num_cameras = 1

    def load(self, quant: str = "none", num_cameras: int = 1, **kwargs: Any) -> None:
        """
        Loads the OpenVLA checkpoint via `wrapper_utils.load_openvla_model`.

        Args:
            quant: quantization out of "none", "8bit", "4bit".
            num_cameras: must be 1 -- base OpenVLA has no wrist-camera input path.
            **kwargs: Unused; present for interface compatibility.

        Raises:
            ValueError: if `quant` is unsupported, or `num_cameras != 1`.
        """
        if quant not in ("none", "8bit", "4bit"):
            raise ValueError(
                f"Unsupported quant '{quant}'. Must be 'none', '8bit', or '4bit'."
            )
        if num_cameras != 1:
            raise ValueError(
                "Base OpenVLA only supports a single (agentview) camera; "
                f"got num_cameras={num_cameras}."
            )

        # patch official OpenVLA assumption that cwd == openvla/
        os.chdir(_OPENVLA_REPO_ROOT)

        self._quant = quant
        self._num_cameras = num_cameras

        torch.cuda.reset_peak_memory_stats()

        register_openvla_auto_classes()
        self._processor = AutoProcessor.from_pretrained(
            self._checkpoint, trust_remote_code=True
        )
        self._vla = load_openvla_model(self._checkpoint, quant)

        self._unnorm_key = resolve_unnorm_key(
            self._unnorm_key, self._vla.norm_stats.keys()
        )
        assert self._unnorm_key in self._vla.norm_stats, (
            f"unnorm_key '{self._unnorm_key}' not found in checkpoint norm_stats "
            f"({list(self._vla.norm_stats.keys())})"
        )

    def reset(self) -> None:
        """
        No-op: base OpenVLA is closed-loop (one action per query, K=1), stateless between
        calls. Implemented to satisfy the PolicyWrapper contract.
        """

    def predict(
        self, observation: Dict[str, Any], task_description: str
    ) -> PolicyOutput:
        """
        Runs one OpenVLA inference call, returning a single action.

        Resize-to-224 + center-crop (model-specific preprocessing) happens inside
        `prepare_openvla_inputs`, excluded from the timed region. Only `predict_action`
        itself is timed, bracketed by `torch.cuda.synchronize()` per CLAUDE.md timing rules.

        Args:
            observation: Dict with key "full_image" (H,W,3 uint8, already 180-degree
                rotated), at LIBERO env resolution (NOT yet resized for the policy).
            task_description: natural-language task instruction.

        Returns:
            PolicyOutput with actions of shape (1, ACTION_DIM), is_chunked=False, and the
            model-only latency in seconds.
        """
        inputs = prepare_openvla_inputs(
            self._processor, observation["full_image"], task_description, self._quant
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            action = self._vla.predict_action(
                **inputs, unnorm_key=self._unnorm_key, do_sample=False
            )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - t0

        actions = np.asarray(action, dtype=np.float32).reshape(1, -1)
        return PolicyOutput(actions=actions, is_chunked=False, raw_latency_s=latency_s)

    def get_metadata(self) -> PolicyMetadata:
        """
        Returns fully-populated metadata for the currently loaded checkpoint.
        """
        if self._vla is None:
            raise RuntimeError("get_metadata() called before load()")
        num_params = sum(p.numel() for p in self._vla.parameters())
        return PolicyMetadata(
            model_id="openvla",
            variant=self._checkpoint,
            num_params=num_params,
            quant=self._quant,
            num_cameras=self._num_cameras,
            action_chunk_size=1,
            attn_impl="sdpa",
            extra={"unnorm_key": self._unnorm_key},
        )

    def get_vram_footprint_gb(self) -> float:
        """
        Returns peak allocated VRAM (GB) observed since the last reset_peak_memory_stats().
        """
        return torch.cuda.max_memory_allocated() / 1e9
