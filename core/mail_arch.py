"""
Porting for MaIL to mamba_ssm version > 2.0 (to exploit sm_120 GPU operations).
Only difference from `mail/`: `_install_mamba_block_shim` below patches around a
mamba_ssm 1.x vs 2.x API difference so mail/'s unmodified code still imports. 
Everything else matches the original`mail/` exactly.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

warnings.filterwarnings(
    "ignore",
    message=r"The parameter 'pretrained' is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Arguments other than a weight enum or `None` for 'weights' are deprecated.*",
    category=UserWarning,
)

_MAIL_REPO_ROOT = Path(__file__).resolve().parent.parent / "mail"


def _install_mamba_block_shim() -> None:
    """mamba_ssm 2.x moved `Block` and added a required `mlp_cls` arg; mail/'s mamba.py
    still does `from mamba_ssm.modules.mamba_simple import Block` (1.x API). Patch a
    subclass with `mlp_cls=nn.Identity` (no MLP in MaIL's blocks) back onto that path."""
    import mamba_ssm.modules.mamba_simple as mamba_simple
    from mamba_ssm.modules.block import Block as _Block

    class _ShimBlock(_Block):
        def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
        ):
            super().__init__(
                dim,
                mixer_cls,
                mlp_cls=nn.Identity,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )

    mamba_simple.Block = _ShimBlock


def _ensure_mail_importable() -> None:
    if str(_MAIL_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_MAIL_REPO_ROOT))
    _install_mamba_block_shim()


@dataclass(frozen=True)
class MailArchConfig:
    """Per-suite architecture hyperparameters, matching mail/'s configs."""

    obs_seq: int
    action_seq: int
    n_layer_encoder: int = 10
    n_layer_decoder: int = 10
    d_state: int = 8
    enc_conv: int = 2
    dec_conv: int = 2
    embed_dim: int = 128
    diff_steps: int = 16
    action_dim: int = 7
    image_size: int = 128

    @property
    def window_size(self) -> int:
        return self.obs_seq + self.action_seq - 1


SUITE_ARCH_CONFIG: Dict[str, MailArchConfig] = {
    "spatial": MailArchConfig(obs_seq=5, action_seq=5, enc_conv=2, dec_conv=2),
    "object": MailArchConfig(obs_seq=5, action_seq=5, enc_conv=2, dec_conv=2),
    "goal": MailArchConfig(obs_seq=5, action_seq=5, enc_conv=2, dec_conv=2),
    "long": MailArchConfig(obs_seq=5, action_seq=10, enc_conv=4, dec_conv=4),
}


class MailPolicy(nn.Module):
    """obs_encoder (ResNet18+SpatialSoftmax) + DDPM(DiffusionMambaCross) action head,
    same as mail/'s `MultiImageObsEncoder` + `Diffusion(DiffusionMambaCross(...))`."""

    def __init__(
        self, arch: MailArchConfig, num_cameras: int, device: str, language: bool = False
    ) -> None:
        super().__init__()
        _ensure_mail_importable()
        import hydra
        from omegaconf import OmegaConf

        self.arch = arch
        self.num_cameras = num_cameras
        self.device = device
        self.language = language
        state_dim = 128 * num_cameras

        if language:
            from transformers import AutoModel, AutoTokenizer

            self.lang_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.language_encoder = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
            self.language_encoder.requires_grad_(False)
            self.language_encoder.eval()
            self.language_encoder.to(device)
            self.lang_linear = nn.Linear(512, state_dim).to(device)

        obs_keys = {"agentview_rgb": {"shape": [3, arch.image_size, arch.image_size], "type": "rgb"}}
        if num_cameras == 2:
            obs_keys["eye_in_hand_rgb"] = {"shape": [3, arch.image_size, arch.image_size], "type": "rgb"}

        obs_encoder_cfg = OmegaConf.create(
            {
                "_target_": "agents.module.vision.multi_image_obs_encoder.MultiImageObsEncoder",
                "shape_meta": {"obs": obs_keys},
                "rgb_model": {
                    "_target_": "agents.module.vision.get_model.get_resnet",
                    "input_shape": [3, arch.image_size, arch.image_size],
                    "output_size": 128,
                },
                "resize_shape": None,
                "random_crop": False,
                "use_group_norm": True,
                "share_rgb_model": False,
                "imagenet_norm": True,
            }
        )

        def mixer_cfg(n_layer: int, d_conv: int):
            return OmegaConf.create(
                {
                    "_target_": "agents.models.oc_ddpm.mamba.MixerModel",
                    "ssm_cfg": {"d_state": arch.d_state, "d_conv": d_conv, "expand": 2},
                    "d_model": arch.embed_dim,
                    "n_layer": n_layer,
                    "norm_epsilon": 1e-5,
                    "rms_norm": False,
                    "fused_add_norm": False,
                    "residual_in_fp32": False,
                    "device": device,
                }
            )

        net_cfg = OmegaConf.create(
            {
                "_target_": "agents.models.oc_ddpm.mamba_cross.DiffusionMambaCross",
                "_recursive_": False,
                "state_dim": state_dim,
                "action_dim": arch.action_dim,
                "goal_conditioned": False,
                "goal_seq_len": 10,
                "obs_seq_len": arch.obs_seq * 2 if language else arch.obs_seq,
                "action_seq_len": arch.action_seq,
                "embed_pdrob": 0,
                "embed_dim": arch.embed_dim,
                "device": device,
                "linear_output": True,
                "encoder": mixer_cfg(arch.n_layer_encoder, arch.enc_conv),
                "mamba": mixer_cfg(arch.n_layer_decoder, arch.dec_conv),
            }
        )

        diffusion_cfg = OmegaConf.create(
            {
                "_target_": "agents.models.oc_ddpm.diffusion_policy.Diffusion",
                "_recursive_": False,
                "state_dim": state_dim,
                "action_dim": arch.action_dim,
                "beta_schedule": "cosine",
                "n_timesteps": arch.diff_steps,
                "loss_type": "l2",
                "clip_denoised": True,
                "predict_epsilon": True,
                "device": device,
                "diffusion_x": False,
                "diffusion_x_M": 10,
                "model": net_cfg,
            }
        )

        self.obs_encoder: nn.Module = hydra.utils.instantiate(obs_encoder_cfg).to(device)
        self.diffusion: nn.Module = hydra.utils.instantiate(diffusion_cfg).to(device)

    def set_action_bounds(self, y_low: torch.Tensor, y_high: torch.Tensor) -> None:
        """Required before sampling; `Diffusion.p_mean_variance` clamps to these bounds."""
        self.diffusion.min_action = y_low.to(self.device)
        self.diffusion.max_action = y_high.to(self.device)

    def train(self, mode: bool = True) -> "MailPolicy":
        """Keeps the frozen CLIP text encoder in eval mode (its transformer blocks have
        dropout) regardless of caller, since it never trains (see `_encode_language`)."""
        super().train(mode)
        if self.language:
            self.language_encoder.eval()
        return self

    def _encode_obs(self, agentview: torch.Tensor, wrist: torch.Tensor | None) -> torch.Tensor:
        b, t = agentview.shape[:2]
        obs_dict = {"agentview_rgb": agentview.reshape(b * t, *agentview.shape[2:])}
        if self.num_cameras == 2:
            assert wrist is not None
            obs_dict["eye_in_hand_rgb"] = wrist.reshape(b * t, *wrist.shape[2:])
        feat = self.obs_encoder(obs_dict)
        return feat.view(b, t, -1)

    def _encode_language(self, language: str | Sequence[str] | torch.Tensor) -> torch.Tensor:
        """language: either raw instruction(s) (a single str or one per batch element,
        live-encoded through the frozen CLIP text encoder -- the only option at eval time,
        see wrappers/mail_wrapper.py), or a precomputed (B, 1, 512)/(B, 512) CLIP embedding
        batch as returned per-window by `LanguageMultiTaskDataset` (mail/'s own
        `task_embeddings/<suite>.pkl`, reused instead of re-encoding during training).
        Returns (B, state_dim). Mirrors mail/'s DiffusionLanguagePolicy: CLIP frozen
        (no_grad), followed by a trainable linear projection (kept outside no_grad so it
        receives gradients)."""
        if isinstance(language, torch.Tensor):
            text_features = language.to(self.device)
            if text_features.dim() == 3:
                text_features = text_features.squeeze(1)
        else:
            texts = [language] if isinstance(language, str) else list(language)
            with torch.no_grad():
                tokens = self.lang_tokenizer(
                    text=texts,
                    add_special_tokens=True,
                    max_length=25,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                ).to(self.device)
                text_features = self.language_encoder.get_text_features(**tokens)
        return self.lang_linear(text_features)

    def _prepend_language(
        self, obs_feat: torch.Tensor, language: str | Sequence[str] | torch.Tensor
    ) -> torch.Tensor:
        """obs_feat: (B, obs_seq, state_dim). Tiles the language embedding across obs_seq
        timesteps and concatenates it in front, matching DiffusionLanguagePolicy.forward's
        `torch.cat([text_embedding, obs], dim=1)` token-concatenation pattern."""
        b, t = obs_feat.shape[:2]
        lang_feat = self._encode_language(language).unsqueeze(1).expand(b, t, -1)
        return torch.cat([lang_feat, obs_feat], dim=1)

    def loss(
        self,
        agentview: torch.Tensor,
        wrist: torch.Tensor | None,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
        language: str | Sequence[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """actions: (B, window_size, action_dim), already normalized. Only the last
        action_seq steps are supervised; the first obs_seq-1 steps only provide
        observation context. mask: (B, window_size) validity mask (1=valid, 0=padded),
        passed as `weights` to `Diffusion.loss`. language: required iff this policy was
        built with `language=True` -- see `_encode_language` for accepted forms."""
        obs_seq = self.arch.obs_seq
        agentview = agentview[:, :obs_seq].contiguous()
        wrist = wrist[:, :obs_seq].contiguous() if wrist is not None else None
        obs_feat = self._encode_obs(agentview, wrist)
        if self.language:
            obs_feat = self._prepend_language(obs_feat, language)
        target = actions[:, obs_seq - 1 :].contiguous()
        if mask is not None:
            weights = mask[:, obs_seq - 1 :].unsqueeze(-1).contiguous()
            return self.diffusion.loss(target, obs_feat, None, weights=weights)
        return self.diffusion.loss(target, obs_feat, None)

    @torch.no_grad()
    def predict(
        self,
        agentview: torch.Tensor,
        wrist: torch.Tensor | None,
        language: str | Sequence[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """agentview/wrist: (1, obs_seq, 3, H, W). Returns (action_seq, action_dim), normalized.
        language: required iff this policy was built with `language=True` -- see
        `_encode_language` for accepted forms."""
        obs_feat = self._encode_obs(agentview, wrist)
        if self.language:
            obs_feat = self._prepend_language(obs_feat, language)
        actions = self.diffusion(obs_feat, None)
        return actions[0]


_FROZEN_STATE_DICT_PREFIX = "language_encoder."


def _strip_frozen_language_encoder(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Excludes the frozen CLIP text encoder's ~63M weights from what gets persisted: they
    never receive gradients (see `MailPolicy._encode_language`) and are trivially
    re-downloaded from HF on load, so including them would roughly double every checkpoint
    write for no benefit."""
    return {k: v for k, v in state_dict.items() if not k.startswith(_FROZEN_STATE_DICT_PREFIX)}


def _scaler_state_dict(scaler: "ActionScaler") -> Dict[str, Any]:
    """mail/'s ActionScaler has no save/load of its own; persist its derived tensors."""
    return {
        "scale_data": scaler.scale_data,
        "y_mean": scaler.y_mean.cpu(),
        "y_std": scaler.y_std.cpu(),
        "y_bounds": scaler.y_bounds_tensor.cpu(),
    }


def _scaler_from_state_dict(state: Dict[str, Any], device: str) -> "ActionScaler":
    from agents.utils import ActionScaler

    scaler = object.__new__(ActionScaler)
    scaler.scale_data = state["scale_data"]
    scaler.device = device
    scaler.y_mean = state["y_mean"].to(device)
    scaler.y_std = state["y_std"].to(device)
    scaler.y_bounds_tensor = state["y_bounds"].to(device)
    return scaler


def save_checkpoint(
    path: Path,
    policy: MailPolicy,
    scaler: "ActionScaler",
    ema_shadow: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _strip_frozen_language_encoder(policy.state_dict()),
            "ema_shadow": _strip_frozen_language_encoder(ema_shadow),
            "scaler": _scaler_state_dict(scaler),
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path: Path, device: str, use_ema: bool = True) -> tuple[MailPolicy, "ActionScaler", dict]:
    """Returns (policy, scaler, meta)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    arch = MailArchConfig(**meta["arch"])
    language = meta.get("language", False)
    policy = MailPolicy(arch, num_cameras=meta["num_cameras"], device=device, language=language)

    state_dict = ckpt["ema_shadow"] if use_ema else ckpt["model_state_dict"]
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected checkpoint keys: {unexpected}"
    assert all(k.startswith(_FROZEN_STATE_DICT_PREFIX) for k in missing), (
        f"missing checkpoint keys other than the excluded frozen language encoder: {missing}"
    )
    policy.eval()

    scaler = _scaler_from_state_dict(ckpt["scaler"], device=device)
    policy.set_action_bounds(scaler.y_bounds_tensor[0], scaler.y_bounds_tensor[1])

    return policy, scaler, meta


def _load_demo_language_embeddings(
    data_dir: List[str], task_suite_name: str, filter_by_attribute: str | None, num_data: int
) -> List[torch.Tensor]:
    """One precomputed (1, 512) CLIP embedding per demo index `i`, matching
    `self.actions`/`self.agentview_rgb`'s flat ordering. Reuses mail/'s own
    `task_embeddings/<suite>.pkl` (built by `mail/get_task_embeddings.py`, same CLIP model)
    instead of re-encoding instructions ourselves -- `mail/dataset/multi_task_dataset_goal.py`
    is the vendored dataset paired with these embeddings, but isn't used directly here since
    it hardcodes an absolute cluster path for the pkl location and drops the `mask` field
    `MailPolicy.loss()` needs; the demo-count-per-file loop below (needed to repeat each
    file's embedding once per demo actually loaded from it) replays
    `MultiTaskDataset.__init__`'s same filter_by_attribute/sort/[:num_data] slicing, same as
    `multi_task_dataset_goal.py`'s own loop.

    Task name (pkl key) is derived from the filename exactly as
    `multi_task_dataset_goal.py` does (`os.path.basename(f).split(".")[0][:-5]`, i.e. minus
    ".hdf5" and the trailing "_demo") -- NOT via `task.language`, which for libero_10 drops a
    "SCENE" prefix the filename keeps.
    """
    import pickle

    import h5py
    import numpy as np

    pkl_path = _MAIL_REPO_ROOT / "task_embeddings" / f"{task_suite_name}.pkl"
    with open(pkl_path, "rb") as f:
        embedding_by_task_name: Dict[str, torch.Tensor] = pickle.load(f)

    demo_embeddings: List[torch.Tensor] = []
    for file_path in data_dir:
        task_name = os.path.basename(file_path).split(".")[0][:-5]
        embedding = embedding_by_task_name[task_name]

        with h5py.File(file_path, "r") as f:
            if filter_by_attribute is not None:
                demo_keys_list = [
                    elem.decode("utf-8")
                    for elem in np.array(f[f"mask/{filter_by_attribute}"][:])
                ]
            else:
                demo_keys_list = list(f["data"].keys())
            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])
            num_demos_used = len(indices[:num_data])

        demo_embeddings.extend([embedding] * num_demos_used)

    return demo_embeddings


class LanguageMultiTaskDataset:
    """Wraps mail/'s `MultiTaskDataset` (vendored, gitignored -- not modified) to also
    return each window's precomputed task-language embedding (see
    `_load_demo_language_embeddings`), matching `multi_task_dataset_goal.py`'s per-sample
    `task_emb`. Composition instead of subclassing: `MultiTaskDataset.__init__` builds
    everything (including `self.slices`) in one pass with no per-demo hook to extend, so
    wrapping and delegating is simpler than replaying its constructor.
    """

    def __init__(self, *, task_suite_name: str, **dataset_kwargs: Any) -> None:
        _ensure_mail_importable()
        from dataset.multi_task_dataset_aug import MultiTaskDataset

        self._dataset = MultiTaskDataset(**dataset_kwargs)
        self._demo_embeddings = _load_demo_language_embeddings(
            self._dataset.data_dir,
            task_suite_name=task_suite_name,
            filter_by_attribute=dataset_kwargs.get("filter_by_attribute"),
            num_data=dataset_kwargs.get("num_data", 10),
        )
        assert len(self._demo_embeddings) == self._dataset.num_data

    def get_all_actions(self) -> torch.Tensor:
        return self._dataset.get_all_actions()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        agentview_rgb, eye_in_hand_rgb, act, mask = self._dataset[idx]
        i, _start, _end = self._dataset.slices[idx]
        return agentview_rgb, eye_in_hand_rgb, act, mask, self._demo_embeddings[i]
