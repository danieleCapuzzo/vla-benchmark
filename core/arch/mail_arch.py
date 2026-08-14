"""Ports MaIL to mamba_ssm version > 2.0 (to exploit sm_120 GPU operations).

Only difference from `mail/`: `_install_mamba_block_shim` below patches around a
mamba_ssm 1.x vs 2.x API difference so mail/'s unmodified code still imports.
Everything else matches the original `mail/` exactly.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from config.paths import MAIL_DIR

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

def _install_mamba_block_shim() -> None:
    """
    Patches mamba_ssm's `Block` to restore the pre-2.x constructor signature.
    Mamba_ssm 2.x moved `Block` and added a required `mlp_cls` arg. Installs a
    subclass with `mlp_cls=nn.Identity` (no MLP in MaIL's blocks).
    """
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
    if str(MAIL_DIR) not in sys.path:
        sys.path.insert(0, str(MAIL_DIR))
    _install_mamba_block_shim()


@dataclass(frozen=True)
class MailArchConfig:
    """
    Per-suite architecture hyperparameters, matching mail/'s configs.
    """

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
    """
    MaIL policy: obs_encoder (ResNet18+SpatialSoftmax) + DDPM(DiffusionMambaCross) action head.
    Matches mail/'s `MultiImageObsEncoder` + `Diffusion(DiffusionMambaCross(...))`.
    """

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
        """
        Sets the action bounds `Diffusion.p_mean_variance` clamps to during sampling.
        Args:
            y_low: Per-dimension lower bound.
            y_high: Per-dimension upper bound.
        """
        self.diffusion.min_action = y_low.to(self.device)
        self.diffusion.max_action = y_high.to(self.device)

    def train(self, mode: bool = True) -> "MailPolicy":
        """
        Sets training mode, keeping the frozen CLIP text encoder in eval mode.
        Args:
            mode: True for training mode, False for evaluation mode.

        Returns:
            This module.
        """
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
        """
        Encodes language into the state embedding space.
        Mirrors mail/'s DiffusionLanguagePolicy: CLIP frozen (no_grad), followed by a
        trainable linear projection.
        Args:
            language: Either raw instruction(s), or a precomputed (B, 1, 512)/(B, 512) CLIP
                embedding batch as returned per-window by `LanguageMultiTaskDataset`.
        Returns:
            (B, state_dim) language embedding.
        """
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
        """
        Prepends the encoded language embedding to the observation feature sequence.
        Args:
            obs_feat: (B, obs_seq, state_dim) observation features.
            language: See `_encode_language`.
        Returns:
            (B, obs_seq + 1, state_dim) sequence with the language embedding prepended.
        """
        b, t = obs_feat.shape[:2]
        lang_feat = self._encode_language(language).unsqueeze(1).expand(b, t, -1)
        return torch.cat([lang_feat, obs_feat], dim=1)

    def loss(
        self, agentview: torch.Tensor, wrist: torch.Tensor | None,
        actions: torch.Tensor, mask: torch.Tensor | None = None,
        language: str | Sequence[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes the diffusion training loss for a batch of observation-action windows.
        Only the last action_seq steps are supervised; the first obs_seq-1 steps only
        provide observation context.
        Args:
            agentview: (B, window_size, 3, H, W) agent-view images.
            wrist: (B, window_size, 3, H, W) wrist-camera images, or None if unused.
            actions: (B, window_size, action_dim) already-normalized actions.
            mask: (B, window_size) validity mask (1=valid, 0=padded), or None.
            language: See `_encode_language`, or None if this policy is not language-conditioned.
        Returns:
            Scalar diffusion loss.
        """
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
        self, agentview: torch.Tensor, wrist: torch.Tensor | None,
        language: str | Sequence[str] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Samples an action sequence from the diffusion policy.
        Args:
            agentview: (1, obs_seq, 3, H, W) agent-view images.
            wrist: (1, obs_seq, 3, H, W) wrist-camera images, or None if unused.
            language: Required iff this policy was built with `language=True` -- see
                `_encode_language` for accepted forms.
        Returns:
            (action_seq, action_dim) normalized action sequence.
        """
        obs_feat = self._encode_obs(agentview, wrist)
        if self.language:
            obs_feat = self._prepend_language(obs_feat, language)
        actions = self.diffusion(obs_feat, None)
        return actions[0]


_FROZEN_STATE_DICT_PREFIX = "language_encoder."
def _strip_frozen_language_encoder(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Excludes the frozen CLIP text encoder's weights from a state dict before persisting.
    Args:
        state_dict: A model or EMA state dict.
    Returns:
        A copy of `state_dict` with all frozen language-encoder keys removed.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith(_FROZEN_STATE_DICT_PREFIX)}


def _scaler_state_dict(scaler: "ActionScaler") -> Dict[str, Any]:
    """
    Builds a serializable state dict for an ActionScaler.
    mail/'s ActionScaler has no save/load of its own; this persists its derived tensors.
    Args:
        scaler: The scaler to serialize.
    Returns:
        A dict of the scaler's derived tensors and config, ready for `torch.save`.
    """
    return {
        "scale_data": scaler.scale_data,
        "y_mean": scaler.y_mean.cpu(),
        "y_std": scaler.y_std.cpu(),
        "y_bounds": scaler.y_bounds_tensor.cpu(),
    }


def _scaler_from_state_dict(state: Dict[str, Any], device: str) -> "ActionScaler":
    """
    Reconstructs an ActionScaler from a state dict produced by `_scaler_state_dict`.
    Args:
        state: A state dict as returned by `_scaler_state_dict`.
        device: Device to load the scaler's tensors onto.
    Returns:
        The reconstructed scaler.
    """
    from agents.utils import ActionScaler

    scaler = object.__new__(ActionScaler)
    scaler.scale_data = state["scale_data"]
    scaler.device = device
    scaler.y_mean = state["y_mean"].to(device)
    scaler.y_std = state["y_std"].to(device)
    scaler.y_bounds_tensor = state["y_bounds"].to(device)
    return scaler


def save_checkpoint(
    path: Path, policy: MailPolicy, scaler: "ActionScaler",
    ema_shadow: Dict[str, torch.Tensor], meta: Dict[str, Any],
) -> None:
    """
    Saves a policy, scaler, and EMA shadow weights to a checkpoint file.
    Args:
        path: Destination file path; parent directories are created if missing.
        policy: The policy whose weights to save.
        scaler: The action scaler to save alongside the policy.
        ema_shadow: EMA shadow weights, keyed like `policy.state_dict()`.
        meta: Arbitrary metadata to persist (e.g. arch config, num_cameras, language flag).
    """
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
    """
    Loads a policy, scaler, and metadata from a checkpoint file.
    Args:
        path: Checkpoint file path, as written by `save_checkpoint`.
        device: Device to load the policy and scaler onto.
        use_ema: If True, load the EMA shadow weights; otherwise load the raw model weights.
    Returns:
        A (policy, scaler, meta) tuple.
    """
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
