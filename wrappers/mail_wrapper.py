"""
Adapter for MambaImitationLearning MaIL (see https://github.com/ALRhub/MaIL) 
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from config.constants import MAIL_IMAGE_SIZE, NUM_CAMERAS_BY_MODEL
from config.paths import REPO_ROOT
from core.core_utils import env_gripper_to_rlds, resize_frame
from core.factory import PolicyFactory
from wrappers.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper
from core.arch.mail_arch import load_checkpoint


def _to_model_image(image: np.ndarray) -> torch.Tensor:
    """
    uint8 HWC (any size) -> float32 CHW in [0, 1] at MAIL_IMAGE_SIZE.
    """
    resized = resize_frame(image, MAIL_IMAGE_SIZE)
    return torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0


@PolicyFactory.register("mail")
class MailWrapper(PolicyWrapper):
    """Adapter pattern: adapts MaIL's DDPM-Mamba policy to the PolicyWrapper contract."""

    def __init__(self, checkpoint: str, unnorm_key: str, language: bool = False) -> None:
        """
        Args:
            checkpoint: Path to a `train_mail.py` checkpoint (.pt) for the target suite.
            unnorm_key: Unused but kept for constructor signature parity
            language:   When True, `predict()` feeds `task_description` into the policy.
                        Must match how the checkpoint was trained; `load()` raises if it doesn't.
        """
        self._checkpoint = checkpoint
        self._language = language
        self._policy = None
        self._scaler = None
        self._meta: Dict[str, Any] = {}
        self._quant = "none"

        self._agentview_ctx: deque = deque()
        self._wrist_ctx: deque = deque()
        self._action_cache: np.ndarray | None = None
        self._action_counter = 0

    def load(
        self,
        quant: str = "none", *, path: str | Path | None = None,
        trunk_path: str | Path | None = None,
        head_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Loads a MambaIL checkpoint (policy + scaler + meta, single file).
        Args:
            quant: Must be "none" as MaIL has no quantized loading path.
            path: Checkpoint path; overrides the checkpoint set in `__init__` when given.
                None reuses the existing checkpoint.
            trunk_path: Unsupported -- MaIL has no trunk/head split (obs_encoder and
                diffusion head are trained end-to-end as one unit).
            head_path: Unsupported, see `trunk_path`.
            **kwargs: Unused; present for interface compatibility.
        Raises:
            ValueError: if `quant != "none"`, `trunk_path`/`head_path` is given, or the
                checkpoint wasn't trained with the 2-camera (agentview + wrist)
                configuration or the language-conditioning setting this wrapper requires.
        """
        if quant != "none":
            raise ValueError(f"MaIL has no quantized loading path; got quant='{quant}'.")
        if trunk_path is not None or head_path is not None:
            raise ValueError("MailWrapper has no trunk/head split; pass `path` only.")
        if path is not None:
            self._checkpoint = str(path)
        self._quant = quant

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.reset_peak_memory_stats()
        checkpoint_path = Path(self._checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        self._policy, self._scaler, self._meta = load_checkpoint(
            checkpoint_path, device=device, use_ema=True
        )

        ckpt_num_cameras = self._meta["num_cameras"]
        if ckpt_num_cameras != NUM_CAMERAS_BY_MODEL["mail"]:
            raise ValueError(
                f"Checkpoint '{self._checkpoint}' was trained with num_cameras="
                f"{ckpt_num_cameras}; MailWrapper requires "
                f"num_cameras={NUM_CAMERAS_BY_MODEL['mail']}."
            )

        ckpt_language = bool(self._meta.get("language", False))
        if ckpt_language != self._language:
            raise ValueError(
                f"Checkpoint '{self._checkpoint}' was trained with language="
                f"{ckpt_language}; MailWrapper was constructed with language={self._language}."
            )

    def reset(self) -> None:
        """
        Clears the observation window and forces a fresh model call on the next `predict()``.
        """
        self._agentview_ctx.clear()
        self._wrist_ctx.clear()
        self._action_cache = None
        self._action_counter = 0

    def predict(self, observation: Dict[str, Any], task_description: str) -> PolicyOutput:
        """
        Runs one MaIL step with action chunking implemented with an 
        internal cache, the network is invoked every `action_seq` calls.
        Args:
            observation: Dict with "raw_obs" (the raw LIBERO env observation dict).
            task_description: Natural-language task instruction. Used only if this
                checkpoint was trained with `--language` (see `load()`); ignored otherwise.
        Returns:
            PolicyOutput with actions of shape (1, ACTION_DIM), is_chunked=False.
        """
        raw_obs = observation["raw_obs"]
        agentview = _to_model_image(raw_obs["agentview_image"])
        self._agentview_ctx.append(agentview)
        arch = self._policy.arch
        while len(self._agentview_ctx) > arch.obs_seq:
            self._agentview_ctx.popleft()

        wrist = _to_model_image(raw_obs["robot0_eye_in_hand_image"])
        self._wrist_ctx.append(wrist)
        while len(self._wrist_ctx) > arch.obs_seq:
            self._wrist_ctx.popleft()
        wrist_seq = torch.stack(list(self._wrist_ctx), dim=0).unsqueeze(0).to(
            self._policy.device
        )

        agentview_seq = torch.stack(list(self._agentview_ctx), dim=0).unsqueeze(0).to(
            self._policy.device
        )

        latency_s = 0.0
        if self._action_cache is None or self._action_counter >= len(self._action_cache):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            normalized_actions = self._policy.predict(
                agentview_seq, wrist_seq, language=task_description if self._language else None
            )
            torch.cuda.synchronize()
            latency_s = time.perf_counter() - t0

            self._action_cache = (
                self._scaler.inverse_scale_output(normalized_actions).cpu().numpy()
            )
            self._action_counter = 0

        action = self._action_cache[self._action_counter]
        self._action_counter += 1

        # convert MaIL's raw env-convention gripper (-1=open/+1=close) to the
        # RLDS convention the evaluator expects
        action = env_gripper_to_rlds(action)

        return PolicyOutput(
            actions=action.reshape(1, -1).astype(np.float32),
            is_chunked=False,
            raw_latency_s=latency_s,
        )

    def get_metadata(self) -> PolicyMetadata:
        """
        Returns fully-populated metadata for the currently loaded checkpoint.
        """
        if self._policy is None:
            raise RuntimeError("get_metadata() called before load()")
        num_params = sum(p.numel() for p in self._policy.parameters())
        return PolicyMetadata(
            model_id="mail",
            variant=f"ddpm_encdec_{self._meta.get('suite', '?')}",
            num_params=num_params,
            quant=self._quant,
            num_cameras=NUM_CAMERAS_BY_MODEL["mail"],
            action_chunk_size=self._policy.arch.action_seq,     # NOTE: used to compute effective_aps
            attn_impl="n/a",                                    # Mamba SSM, no attention
            extra={
                "replan_interval": self._policy.arch.action_seq,
                "num_demos": self._meta.get("num_demos"),
                "train_epoch": self._meta.get("epoch"),
                "language": self._language,
            },
        )

    def get_vram_footprint_gb(self) -> float:
        """
        Returns peak allocated VRAM (GB) observed since the last reset_peak_memory_stats().
        """
        return torch.cuda.max_memory_allocated() / 1e9
