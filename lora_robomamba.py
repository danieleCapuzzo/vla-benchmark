#!/usr/bin/env python
"""
Standalone online LoRA fine-tuning of RoboMamba (CLIP ViT-L/14 + Mamba-2.8B) on a single LIBERO
suite, targeting 16 GB VRAM GPUs. Independent of `train_robomamba.py`'s frozen-trunk / cached-
feature pipeline -- this script updates the trunk itself via LoRA, every step, online.

Hyperparameters live in `RoboMambaLoraConfig` below -- edit its defaults directly to change them,
there is no CLI override for them. Run in the `bench-env` conda env:
    python lora_robomamba.py --suite spatial
    python lora_robomamba.py --suite spatial --resume

Checkpointing happens every `CHECKPOINT_EVERY_STEPS` optimizer steps (mid-epoch) and at the end of
every epoch: `trained/robomamba_lora/<suite>/epoch_<N>.pt` (epoch boundaries) or `step_<N>.pt`
(mid-epoch), with `last.pt` (a copy of the latest, either kind) as the `--resume` target. A
mid-epoch resume reconstructs the same epoch's DataLoader (same seed => same batch order) and
fast-forwards past the batches already consumed. `train_log.csv` in the same directory is
appended every optimizer step.

Architecture: see core/arch/robomamba_arch.py. Data loading: see core/dataproc/libero_dataset.py.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r"`resume_download` is deprecated.*"
)

from config.constants import SUITE_TO_DATASET_DIR
from config.paths import LIBERO_DATASETS
from core.arch.robomamba_arch import (
    ACTION_DIM,
    HIDDEN_DIM,
    RoboMambaHead,
    build_trunk,
    encode_features,
    force_mamba_module_path,
)
from core.dataproc.libero_dataset import LiberoHDF5Dataset, compute_action_bounds, make_collate_fn

CHECKPOINT_EVERY_STEPS = 100


@dataclass(frozen=True)
class RoboMambaLoraConfig:
    """Hyperparameters for one LoRA fine-tuning run. Serialized into every checkpoint and
    re-asserted equal to the current config on `--resume`."""

    suite: str
    data_dir: Path | None = None
    trunk_checkpoint: Path = Path("trained/robomamba/RoboMamba-224-llava-R300-checkpoint.pth")
    # LoRA
    lora_rank: int = 16  # start here; re-run at 32 only if 16 underfits
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_vit: bool = False  # ViT-L frozen for the initial experiment
    lora_projector: bool = True
    lora_mamba: bool = True
    # optimization
    epochs: int = 5
    batch_size: int = 8  # fallback 4 if it doesn't fit
    grad_accum: int = 8  # effective batch 64
    lora_lr: float = 5e-4
    head_lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 100  # head-only; LoRA frozen until this many optimizer steps
    # data / io
    augment: bool = True
    num_workers: int = 4
    seed: int = 7
    output_dir: Path = Path("trained/robomamba_lora")


def _lora_targets(trunk: nn.Module, cfg: RoboMambaLoraConfig) -> list[str]:
    """Full module paths of `nn.Linear` layers to adapt, gated by `cfg`'s per-component flags."""
    targets = []
    for name, module in trunk.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.startswith("vision.") and cfg.lora_vit:
            targets.append(name)
        elif name.startswith("projector.") and cfg.lora_projector:
            targets.append(name)
        elif name.startswith("llm.") and cfg.lora_mamba and not name.endswith(("dt_proj", "lm_head")):
            targets.append(name)
    return targets


def _set_lora_trainable(model: nn.Module, trainable: bool) -> None:
    """Toggles `requires_grad` on every LoRA adapter parameter (head-only warmup switch)."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(trainable)


def _check_gradient_liveness(model: nn.Module, cfg: RoboMambaLoraConfig) -> None:
    """Asserts a just-completed backward() produced non-zero gradients for every adapted
    component -- regression check for HF Mamba's fast-path bypass (see `force_mamba_module_path`)."""
    checks = []
    if cfg.lora_vit:
        checks.append("vision.model.blocks.0.attn.qkv")
    if cfg.lora_projector:
        checks.append("projector.projector.0")
    if cfg.lora_mamba:
        checks += [
            "llm.mamba.backbone.layers.0.mixer.in_proj",
            "llm.mamba.backbone.layers.0.mixer.x_proj",
            "llm.mamba.backbone.layers.0.mixer.out_proj",
        ]
    params = dict(model.named_parameters())
    for name in checks:
        # lora_B, not lora_A: PEFT zero-inits lora_B, so lora_A's grad is legitimately zero on
        # the first step and useless as a liveness signal.
        key = f"base_model.model.{name}.lora_B.default.weight"
        p = params.get(key)
        assert p is not None, f"expected LoRA param {key} not found -- target_modules mismatch?"
        assert p.grad is not None, f"{key}.grad is None -- LoRA gradient did not flow (fast-path bypass?)"
        assert p.grad.abs().max().item() > 0, f"{key}.grad is all-zero"
    print(f"[{cfg.suite}] gradient liveness check passed for {len(checks)} adapted components")


def _save_checkpoint(path: Path, payload: dict) -> None:
    """Atomic checkpoint write (temp file + os.replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _save_named_checkpoint(run_dir: Path, name: str, payload: dict) -> Path:
    """Writes `<name>.pt` and refreshes `last.pt` (the resume target) to match."""
    ckpt_path = run_dir / f"{name}.pt"
    _save_checkpoint(ckpt_path, payload)
    last_tmp = run_dir / "last.pt.tmp"
    shutil.copy2(ckpt_path, last_tmp)
    os.replace(last_tmp, run_dir / "last.pt")
    return ckpt_path


def parse_args() -> tuple[str, bool, int | None]:
    """Parses the run-mode CLI: which suite, whether to resume, and an optional debug step cap.
    All hyperparameters come from `RoboMambaLoraConfig`'s defaults, not from CLI flags."""
    p = argparse.ArgumentParser(description="Online LoRA fine-tuning of RoboMamba on one LIBERO suite")
    p.add_argument("--suite", required=True, choices=list(SUITE_TO_DATASET_DIR))
    p.add_argument("--resume", action="store_true", help="Resume from <output-dir>/<suite>/last.pt.")
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Debug/smoke-test cap on optimizer steps; stops mid-epoch without checkpointing.",
    )
    args = p.parse_args()
    return args.suite, args.resume, args.max_steps


def train(cfg: RoboMambaLoraConfig, resume: bool, max_steps: int | None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    dataset_dir_name = SUITE_TO_DATASET_DIR[cfg.suite]
    data_dir = cfg.data_dir or (LIBERO_DATASETS / f"{dataset_dir_name}_regen")
    if not data_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {data_dir}")
    run_dir = cfg.output_dir / cfg.suite
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    last_ckpt_path = run_dir / "last.pt"

    resume_ckpt = None
    if resume:
        if not last_ckpt_path.exists():
            raise RuntimeError(f"--resume given but no checkpoint at {last_ckpt_path}")
        resume_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
        saved_cfg = RoboMambaLoraConfig(**resume_ckpt["config"])
        assert saved_cfg == cfg, f"resume config mismatch:\n  saved: {saved_cfg}\n  given: {cfg}"
        q01, q99 = np.asarray(resume_ckpt["action_q01"]), np.asarray(resume_ckpt["action_q99"])
        start_epoch = resume_ckpt["epoch"]
        global_step = resume_ckpt["global_step"]
        skip_micro_steps = resume_ckpt.get("micro_step_in_epoch", 0)
    else:
        q01, q99 = compute_action_bounds(data_dir, cfg.suite)
        start_epoch = 0
        global_step = 0
        skip_micro_steps = 0
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "epoch", "loss", "lr_lora", "lr_head"])

    print(
        f"[{cfg.suite}] action bounds q01={q01.round(4)} q99={q99.round(4)}\n"
        f"[{cfg.suite}] lora r={cfg.lora_rank} alpha={cfg.lora_alpha} "
        f"vit={cfg.lora_vit} projector={cfg.lora_projector} mamba={cfg.lora_mamba}\n"
        f"[{cfg.suite}] epochs={cfg.epochs} batch={cfg.batch_size} grad_accum={cfg.grad_accum} "
        f"effective_batch={cfg.batch_size * cfg.grad_accum} warmup_steps={cfg.warmup_steps}"
    )

    print(f"[{cfg.suite}] loading trunk from {cfg.trunk_checkpoint}...")
    t0 = time.perf_counter()
    trunk = build_trunk(cfg.trunk_checkpoint, device=device, dtype=torch.bfloat16)
    trunk.llm.mamba.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trunk.vision.model.set_grad_checkpointing(True)
    print(f"[{cfg.suite}] trunk loaded in {time.perf_counter() - t0:.1f}s (grad checkpointing on)")

    lora_config = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=_lora_targets(trunk, cfg), init_lora_weights="gaussian",
    )
    model = get_peft_model(trunk, lora_config)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()  # adapter master weights stay fp32
    model.print_trainable_parameters()

    model.train()
    n_patched = force_mamba_module_path(model)  # re-patch: .train() above flipped mixers back
    assert n_patched > 0, "no MambaMixer modules found -- trunk structure may have changed"

    head = RoboMambaHead(in_dim=HIDDEN_DIM).to(device=device, dtype=torch.float32)

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    clip_params = lora_params + list(head.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": cfg.lora_lr}, {"params": head.parameters(), "lr": cfg.head_lr}],
        betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay,
    )

    if resume_ckpt is not None:
        set_peft_model_state_dict(model, resume_ckpt["adapter_state"])
        head.load_state_dict(resume_ckpt["head_state"])
        optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        print(
            f"[{cfg.suite}] resumed from {last_ckpt_path} at epoch {start_epoch}, step {global_step}"
            + (f", {skip_micro_steps} micro-batches into the epoch" if skip_micro_steps else "")
        )

    lora_trainable = global_step >= cfg.warmup_steps
    _set_lora_trainable(model, lora_trainable)
    print(f"[{cfg.suite}] LoRA trainable={lora_trainable} (warmup_steps={cfg.warmup_steps})")

    dataset = LiberoHDF5Dataset(
        data_dir, cfg.suite, augment=cfg.augment, action_q01=q01, action_q99=q99, seed=cfg.seed
    )
    steps_per_epoch = (len(dataset) // cfg.batch_size) // cfg.grad_accum
    print(f"[{cfg.suite}] {len(dataset)} frames  {steps_per_epoch} optimizer steps/epoch")

    checked_liveness = resume_ckpt is not None  # already proven if resuming

    for epoch in range(start_epoch, cfg.epochs):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            num_workers=cfg.num_workers, collate_fn=make_collate_fn(model.tokenizer),
            generator=torch.Generator().manual_seed(cfg.seed + epoch),
        )

        epoch_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_step, last_step_loss = 0, None
        pbar = tqdm(loader, desc=f"[{cfg.suite}] epoch {epoch + 1}/{cfg.epochs}", unit="frame", unit_scale=cfg.batch_size)
        pbar_iter = iter(pbar)
        stop_early = False
        if epoch == start_epoch and skip_micro_steps:
            # same seed => same batch order, so replaying and discarding reproduces the exact
            # remaining stream instead of re-training on already-seen batches
            for _ in range(skip_micro_steps):
                next(pbar_iter)
            micro_step = skip_micro_steps
            tqdm.write(f"[{cfg.suite}] fast-forwarded {skip_micro_steps} micro-batches into epoch {epoch + 1}")
        for batch in pbar_iter:
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            actions = batch["actions"].to(device=device, dtype=torch.float32)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = encode_features(model, pixel_values, input_ids, attention_mask)
            pred = head(features.float())
            loss = F.l1_loss(pred, actions) / cfg.grad_accum
            loss.backward()
            micro_step += 1

            if lora_trainable and not checked_liveness:
                # first backward() since LoRA became trainable (warmup may still be running
                # otherwise -- grad=None during warmup is expected, not a bypass)
                _check_gradient_liveness(model, cfg)
                checked_liveness = True

            if micro_step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step == cfg.warmup_steps and not lora_trainable:
                    lora_trainable = True
                    _set_lora_trainable(model, True)
                    tqdm.write(f"[{cfg.suite}] step {global_step}: warmup done, LoRA unfrozen")

                last_step_loss = loss.item() * cfg.grad_accum
                pbar.set_postfix(step=global_step, loss=f"{last_step_loss:.4f}")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        global_step, epoch + 1, last_step_loss,
                        optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"],
                    ])

                if global_step % CHECKPOINT_EVERY_STEPS == 0:
                    payload = {
                        "adapter_state": get_peft_model_state_dict(model),
                        "head_state": head.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,  # still mid this epoch -- resume must redo it, not skip it
                        "global_step": global_step,
                        "micro_step_in_epoch": micro_step,
                        "action_q01": q01,
                        "action_q99": q99,
                        "config": dataclasses.asdict(cfg),
                    }
                    step_path = _save_named_checkpoint(run_dir, f"step_{global_step}", payload)
                    tqdm.write(f"[{cfg.suite}] saved checkpoint: {step_path}")

                if max_steps is not None and global_step >= max_steps:
                    tqdm.write(f"[{cfg.suite}] hit --max-steps={max_steps}, stopping (no checkpoint saved)")
                    stop_early = True
                    break
        if stop_early:
            return

        epoch_time_s = time.perf_counter() - epoch_t0
        print(
            f"[{cfg.suite}] epoch {epoch + 1}/{cfg.epochs} done in {epoch_time_s:.1f}s  "
            f"step {global_step}  last_loss {last_step_loss:.4f}"
        )

        payload = {
            "adapter_state": get_peft_model_state_dict(model),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "micro_step_in_epoch": 0,
            "action_q01": q01,
            "action_q99": q99,
            "config": dataclasses.asdict(cfg),
        }
        epoch_path = _save_named_checkpoint(run_dir, f"epoch_{epoch + 1}", payload)
        print(f"[{cfg.suite}] saved checkpoint: {epoch_path}")

    print(f"[{cfg.suite}] training done: {cfg.epochs} epochs, {global_step} steps. checkpoints in {run_dir}")


if __name__ == "__main__":
    suite, resume, max_steps = parse_args()
    train(RoboMambaLoraConfig(suite=suite), resume, max_steps)
