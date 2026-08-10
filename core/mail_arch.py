"""
Porting for MaIL to mamba_ssm version > 2.0 (to exploit sm_120 GPU operations).
Only difference from `mail/`: `_install_mamba_block_shim` below patches around a
mamba_ssm 1.x vs 2.x API difference so mail/'s unmodified code still imports. 
Everything else matches the original`mail/` exactly.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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

    def __init__(self, arch: MailArchConfig, num_cameras: int, device: str) -> None:
        super().__init__()
        _ensure_mail_importable()
        import hydra
        from omegaconf import OmegaConf

        self.arch = arch
        self.num_cameras = num_cameras
        self.device = device
        state_dim = 128 * num_cameras

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
                "obs_seq_len": arch.obs_seq,
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

    def _encode_obs(self, agentview: torch.Tensor, wrist: torch.Tensor | None) -> torch.Tensor:
        b, t = agentview.shape[:2]
        obs_dict = {"agentview_rgb": agentview.reshape(b * t, *agentview.shape[2:])}
        if self.num_cameras == 2:
            assert wrist is not None
            obs_dict["eye_in_hand_rgb"] = wrist.reshape(b * t, *wrist.shape[2:])
        feat = self.obs_encoder(obs_dict)
        return feat.view(b, t, -1)

    def loss(
        self,
        agentview: torch.Tensor,
        wrist: torch.Tensor | None,
        actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """actions: (B, window_size, action_dim), already normalized. Only the last
        action_seq steps are supervised; the first obs_seq-1 steps only provide
        observation context. mask: (B, window_size) validity mask (1=valid, 0=padded),
        passed as `weights` to `Diffusion.loss`."""
        obs_seq = self.arch.obs_seq
        agentview = agentview[:, :obs_seq].contiguous()
        wrist = wrist[:, :obs_seq].contiguous() if wrist is not None else None
        obs_feat = self._encode_obs(agentview, wrist)
        target = actions[:, obs_seq - 1 :].contiguous()
        if mask is not None:
            weights = mask[:, obs_seq - 1 :].unsqueeze(-1).contiguous()
            return self.diffusion.loss(target, obs_feat, None, weights=weights)
        return self.diffusion.loss(target, obs_feat, None)

    @torch.no_grad()
    def predict(self, agentview: torch.Tensor, wrist: torch.Tensor | None) -> torch.Tensor:
        """agentview/wrist: (1, obs_seq, 3, H, W). Returns (action_seq, action_dim), normalized."""
        obs_feat = self._encode_obs(agentview, wrist)
        actions = self.diffusion(obs_feat, None)
        return actions[0]


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
            "model_state_dict": policy.state_dict(),
            "ema_shadow": ema_shadow,
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
    policy = MailPolicy(arch, num_cameras=meta["num_cameras"], device=device)

    state_dict = ckpt["ema_shadow"] if use_ema else ckpt["model_state_dict"]
    policy.load_state_dict(state_dict)
    policy.eval()

    scaler = _scaler_from_state_dict(ckpt["scaler"], device=device)
    policy.set_action_bounds(scaler.y_bounds_tensor[0], scaler.y_bounds_tensor[1])

    return policy, scaler, meta
