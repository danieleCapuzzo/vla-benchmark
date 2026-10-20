"""
Adapter for RoboMamba (see https://github.com/lmzpai/roboMamba): CLIP ViT-L/14 + Mamba-2.8B
trunk with a LIBERO action-chunking head, evaluated from a single merged checkpoint produced
by `external/roboMamba-libero/merge_weights.py`.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from config.constants import NUM_CAMERAS_BY_MODEL
from config.paths import ROBOMAMBA_LIBERO_SRC
from core.core_utils import denormalize_actions, env_gripper_to_rlds, resize_frame, to_upright
from core.factory import PolicyFactory
from wrappers.policy_wrapper import PolicyMetadata, PolicyOutput, PolicyWrapper

_REPO_ROOT = Path(__file__).resolve().parent.parent

# roboMamba-libero's src/ has no __init__.py 
# and imports as `model.vision`, `model.manip`, etc.
if str(ROBOMAMBA_LIBERO_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOMAMBA_LIBERO_SRC))

# checkpoint format the merged .pt must carry (see merge_weights.py)
_REQUIRED_KEYS = {"state_dict", "tokenizer_json", "tokenizer_kwargs", "mamba_config", "meta"}


def _resolve(path: str | Path) -> Path:
    """Resolves a repo-relative checkpoint path to absolute."""
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _build_model(ckpt: Dict[str, Any], device: str):
    """
    Rebuilds a `LinearManip` (trunk + action head) from a merged checkpoint, fully offline.
    Args:
        ckpt: The loaded checkpoint dict; must carry `_REQUIRED_KEYS`.
        device: Target device for the loaded model.
    Returns:
        A `LinearManip` in eval mode, trunk frozen, action head cast to fp32.
    """
    import model.vision as vision_module
    from accelerate import init_empty_weights
    from model.llm import MambaLLM, mamba_dict
    from model.manip import LinearManip
    from model.vision import Vision
    from transformers import MambaConfig, MambaForCausalLM, PreTrainedTokenizerFast

    meta = ckpt["meta"]

    # tokenizer.json is baked in as a string; PreTrainedTokenizerFast needs a real file path
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(ckpt["tokenizer_json"])
        tokenizer_path = f.name
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, **ckpt["tokenizer_kwargs"])
    finally:
        Path(tokenizer_path).unlink()

    # Vision("CLIP224") hardcodes timm pretrained=True
    # forced off to avoid huge download
    orig_create_model = vision_module.timm.create_model

    def _create_model_untrained(*args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return orig_create_model(*args, **kwargs)

    vision_module.timm.create_model = _create_model_untrained
    try:
        with init_empty_weights():
            vision = Vision(meta["vision_encoder"])

            # bypasses MambaLLM.__init__'s from_pretrained() Hub calls
            llm = MambaLLM.__new__(MambaLLM)
            nn.Module.__init__(llm)
            llm.mamba_type = mamba_dict["mamba-2.8b"]
            llm.mamba = MambaForCausalLM(MambaConfig(**ckpt["mamba_config"]))
            llm.tokenizer = tokenizer
            llm.hidden_size = llm.mamba.config.hidden_size

            model = LinearManip(
                vision, llm, action_chunk=meta["action_chunk"], action_dim=meta["action_dim"]
            )
    finally:
        vision_module.timm.create_model = orig_create_model

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False, assign=True)
    assert not missing, f"checkpoint left params unset: {missing}"
    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
    leftover_meta = [n for n, p in model.named_parameters() if p.is_meta]
    assert not leftover_meta, f"params still on meta device after load: {leftover_meta}"

    model.tokenizer.pad_token = model.tokenizer.eos_token
    model = model.to(device=device)
    model.action_head.to(dtype=torch.float32)  # head trained in fp32; the rest stays bf16
    model.eval()
    model.requires_grad_(False)
    return model


@PolicyFactory.register("robomamba")
class RoboMambaWrapper(PolicyWrapper):
    """Adapter pattern: adapts RoboMamba's merged checkpoint to the PolicyWrapper contract."""

    def __init__(self, checkpoint: str, unnorm_key: str, suite: str | None = None) -> None:
        """
        Args:
            checkpoint: Path to a RoboMamba checkpoint (.pt) for the target suite.
            unnorm_key: Unused but kept for constructor signature parity.
            suite: Suite this wrapper is being evaluated on. If given, `load()` asserts it
                matches the checkpoint's own `meta["suite"]`.
        """
        self._checkpoint = checkpoint
        self._suite = suite
        self._model = None
        self._meta: Dict[str, Any] = {}
        self._quant = "none"
        self._action_q01: np.ndarray | None = None
        self._action_q99: np.ndarray | None = None

    def load(
        self,
        quant: str = "none",
        *,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Loads a full RoboMamba checkpoint (trunk + action head, single file).
        Args:
            quant: Must be "none"; RoboMamba has no quantized loading path.
            path: Checkpoint path; overrides the checkpoint set in `__init__` when given.
                None reuses the existing checkpoint.
            **kwargs: Unused; present for interface compatibility.
        Raises:
            ValueError: if `quant != "none"`.
            AssertionError: if the checkpoint's payload is missing required keys, or its
                `meta["suite"]` disagrees with the suite this wrapper was constructed for.
        """
        if quant != "none":
            raise ValueError(f"RoboMamba has no quantized loading path; got quant='{quant}'.")
        self._quant = quant

        if path is not None:
            self._checkpoint = str(path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.reset_peak_memory_stats()

        ckpt_path = _resolve(self._checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        missing_keys = _REQUIRED_KEYS - ckpt.keys()
        assert not missing_keys, (
            f"{ckpt_path} is missing {missing_keys} -- not a merged checkpoint "
            "(see external/roboMamba-libero/merge_weights.py)"
        )

        self._meta = ckpt["meta"]
        if self._suite is not None:
            assert self._meta["suite"] == self._suite, (
                f"checkpoint suite '{self._meta['suite']}' != requested suite '{self._suite}'"
            )

        self._model = _build_model(ckpt, device=device)
        self._action_q01 = np.asarray(self._meta["action_q01"], dtype=np.float32)
        self._action_q99 = np.asarray(self._meta["action_q99"], dtype=np.float32)

    def reset(self) -> None:
        """
        No-op: the head is stateless (single-frame features, no cross-step history). Kept for
        interface parity with wrappers that do carry episode state.
        """

    def predict(self, observation: Dict[str, Any], task_description: str) -> PolicyOutput:
        """
        Runs one RoboMamba step: trunk forward (language-pooled readout) -> action head.
        Args:
            observation: Dict with "raw_obs" (the raw LIBERO env observation dict).
            task_description: Natural-language task instruction, always used.
        Returns:
            PolicyOutput with actions of shape (action_chunk, 7), is_chunked=True.
        Raises:
            RuntimeError: If `load()` has not been called yet.
        """
        if self._model is None:
            raise RuntimeError("predict() requires load() to have been called first.")
        device = next(self._model.parameters()).device
        model_dtype = self._model.llm.mamba.dtype

        # matches training's preprocessing exactly (see external/roboMamba-libero/src/data/
        # libero.py): upright -> INTER_AREA resize -> [0,1] -> CLIP mean/std, no crop.
        frame = to_upright(observation["raw_obs"]["agentview_image"])
        frame = resize_frame(frame, self._meta["image_size"])
        img = torch.from_numpy(frame).float().div(255.0)
        mean = torch.tensor(self._meta["image_mean"]).view(3, 1, 1)
        std = torch.tensor(self._meta["image_std"]).view(3, 1, 1)
        pixel_values = img.permute(2, 0, 1).sub(mean).div(std)
        pixel_values = pixel_values.unsqueeze(0).to(device=device, dtype=model_dtype)

        prompt = self._meta["prompt_template"].format(instruction=task_description)
        input_ids = self._model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # latency covers the model call only
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            actions = self._model(pixel_values, input_ids)
        torch.cuda.synchronize()
        latency_s = time.perf_counter() - t0

        actions = actions.squeeze(0).detach().cpu().numpy().astype(np.float32)

        # order matters: gripper remap assumes raw LIBERO units, so denormalize first
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
        if self._model is None:
            raise RuntimeError("get_metadata() called before load()")
        num_params = sum(p.numel() for p in self._model.parameters())
        return PolicyMetadata(
            model_id="robomamba",
            variant=f"clip224_mamba2.8b_{self._meta.get('suite', '?')}_epoch{self._meta.get('train_epoch', '?')}",
            num_params=num_params,
            quant=self._quant,
            num_cameras=NUM_CAMERAS_BY_MODEL["robomamba"],
            action_chunk_size=self._meta["action_chunk"],
            attn_impl="n/a",  # Mamba SSM, no attention
            extra={
                "train_epoch": self._meta.get("train_epoch"),
                "train_kind": self._meta.get("train_kind"),
                "lora": self._meta.get("lora"),
                "trunk_checkpoint": self._meta.get("trunk_checkpoint"),
                "hidden_dim": self._meta.get("hidden_dim"),
            },
        )

    def get_vram_footprint_gb(self) -> float:
        """
        Returns peak allocated VRAM (GB) observed since the last reset_peak_memory_stats().
        """
        return torch.cuda.max_memory_allocated() / 1e9
