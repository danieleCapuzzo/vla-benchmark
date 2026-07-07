"""
Adapter for OpenVLA-OFT (see https://github.com/moojink/openvla-oft)

Wraps the official inference pipeline and uses official OFT API
(get_vla, get_processor, get_action_head, get_proprio_projector, get_vla_action, GenerateConfig)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from wrappers.wrapper_utils import OPENVLA_INPUT_RESOLUTION

# patch issue with official OFT harness using package-relative imports
# that only resolve when openvla-oft/ is on sys.path
_OFT_REPO_ROOT = Path(__file__).resolve().parent.parent / "openvla-oft"
if str(_OFT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFT_REPO_ROOT))

from core.factory import PolicyFactory  # noqa: E402
from core.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig  # noqa: E402
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
    resize_image_for_policy,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402


@PolicyFactory.register("oft")
class OftWrapper(PolicyWrapper):
    """
    Adapter pattern: adapts OFT's `get_vla_action` API to the PolicyWrapper contract.
    """

    def __init__(self, checkpoint: str, unnorm_key: str) -> None:
        """
        Creates the wrapper. Model weights are loaded lazily in `load()`.

        Args:
            checkpoint: HF Hub id (or local path) of the OFT checkpoint for the target suite.
            unnorm_key: action un-normalization key matching this checkpoint
                (e.g. "libero_spatial_no_noops").
        """
        self._checkpoint = checkpoint
        self._unnorm_key = unnorm_key
        self._cfg = None
        self._vla = None
        self._processor = None
        self._action_head = None
        self._proprio_projector = None
        self._quant = "none"
        self._num_cameras = 1

    def load(self, quant: str = "none", num_cameras: int = 1, **kwargs: Any) -> None:
        """
        Loads the OFT checkpoint via the official pipeline.

        Args:
            quant: quantization out of "none", "8bit", "4bit".
            num_cameras: 1 (primary, VRAM-constrained) or 2 (ablation, if VRAM permits).
            **kwargs: Unused; present for interface compatibility.

        Raises:
            ValueError: if `quant` is not one of the supported values.
        """
        if quant not in ("none", "8bit", "4bit"):
            raise ValueError(
                f"Unsupported quant '{quant}'. Must be 'none', '8bit', or '4bit'."
            )

        # patch official OFT assumption that cwd == openvla-oft/
        os.chdir(_OFT_REPO_ROOT)

        # set env variables and prepare configuration
        self._quant = quant
        self._num_cameras = num_cameras
        self._num_actions_chunk = NUM_ACTIONS_CHUNK

        self._cfg = GenerateConfig(
            pretrained_checkpoint=self._checkpoint,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=num_cameras,
            use_proprio=True,
            load_in_8bit=(quant == "8bit"),
            load_in_4bit=(quant == "4bit"),
            center_crop=True,
            num_open_loop_steps=NUM_ACTIONS_CHUNK,
            unnorm_key=self._unnorm_key,
        )

        # ensure memory stats are reset
        torch.cuda.reset_peak_memory_stats()

        self._vla = get_vla(self._cfg)
        self._processor = get_processor(self._cfg)
        self._action_head = get_action_head(self._cfg, llm_dim=self._vla.llm_dim)
        self._proprio_projector = get_proprio_projector(
            self._cfg, llm_dim=self._vla.llm_dim, proprio_dim=PROPRIO_DIM
        )

    def reset(self) -> None:
        """
        No-op: the OFT wrapper is stateless between calls.
        The action-chunk queue (open-loop execution of the K=8 chunk between requeries) is
        owned by the evaluator's rollout loop, not by this wrapper, since it is a
        control-loop concern shared by any chunked model. Implemented to satisfy the
        PolicyWrapper contract.
        """

    def predict(
        self, observation: Dict[str, Any], task_description: str
    ) -> PolicyOutput:
        """
        Runs one OFT inference call, returning the full (K, action_dim) action chunk.

        Resize-to-224 (model-specific preprocessing) happens here, excluded from the timed
        region. Only the `get_vla_action` call itself is timed, bracketed by
        `torch.cuda.synchronize()` per doc/ARCHITECTURE.md §5.2 / CLAUDE.md timing rules.

        Args:
            observation: Dict with keys "full_image" (H,W,3 uint8), optionally
                "wrist_image", and "state" (proprioception), at LIBERO env resolution
                (NOT yet resized for the policy).
            task_description: natural-language task instruction.

        Returns:
            PolicyOutput with actions of shape (NUM_ACTIONS_CHUNK, ACTION_DIM), is_chunked=True,
            and the model-only latency in seconds.
        """

        model_obs: Dict[str, Any] = {
            "full_image": resize_image_for_policy(
                observation["full_image"], OPENVLA_INPUT_RESOLUTION
            ),
            "state": observation["state"],
        }
        if "wrist_image" in observation:
            model_obs["wrist_image"] = resize_image_for_policy(
                observation["wrist_image"], OPENVLA_INPUT_RESOLUTION
            )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        action_chunk = get_vla_action(
            self._cfg,
            self._vla,
            self._processor,
            model_obs,
            task_description,
            action_head=self._action_head,
            proprio_projector=self._proprio_projector,
        )
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - t0

        actions = np.asarray(action_chunk, dtype=np.float32)
        return PolicyOutput(actions=actions, is_chunked=True, raw_latency_s=latency_s)

    def get_metadata(self) -> PolicyMetadata:
        """
        Returns fully-populated metadata for the currently loaded checkpoint.
        """
        if self._vla is None:
            raise RuntimeError("get_metadata() called before load()")
        num_params = sum(p.numel() for p in self._vla.parameters())
        return PolicyMetadata(
            model_id="oft",
            variant=self._checkpoint,
            num_params=num_params,
            quant=self._quant,
            num_cameras=self._num_cameras,
            action_chunk_size=self._num_actions_chunk,
            attn_impl="sdpa",
            extra={
                "unnorm_key": self._unnorm_key,
                "num_images_in_input": self._num_cameras,
            },
        )

    def get_vram_footprint_gb(self) -> float:
        """
        Returns peak allocated VRAM (GB) observed since the last reset_peak_memory_stats().
        """
        return torch.cuda.max_memory_allocated() / 1e9
