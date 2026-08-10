#!/usr/bin/env python
"""
Training script and CLI entrypoint for MambaIL-2camera (DDPM, Enc-Dec) on
a single libero suite passed as a CLI argument.

Run in the `bench-env` conda env:
    python train_mail.py --suite spatial
    python train_mail.py --suite long --num-demos 10   # paper's 20%-data setting

Trains on 100% of the demos per task by default (--num-demos, unset = all 50/task),
matching how OpenVLA/OpenVLA-OFT were finetuned. 

Model architecture: imported from mail/ (see core/mail_arch.py)
Dataset: `mail/dataset/multi_task_dataset_aug.py MultiTaskDataset` used directly
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.mail_arch import SUITE_ARCH_CONFIG, MailPolicy, _ensure_mail_importable, save_checkpoint

SUITE_TO_DATASET_DIR = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}

OBS_MODALITIES = {
    "obs": {
        "low_dim": ["ee_ori", "ee_pos", "gripper_states", "joint_states"],
        "rgb": ["agentview_rgb", "eye_in_hand_rgb"],
        "depth": [],
        "scan": [],
    },
    "goal": {"low_dim": [], "rgb": [], "depth": [], "scan": []},
}
MAX_LEN_DATA = 520  # mail/config/benchmark_libero_new.yaml; must exceed every demo's length


class EmaTracker:
    """Minimal exponential moving average over a model's parameters, dict-keyed so the
    shadow can be saved/loaded directly with `nn.Module.load_state_dict`."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(param, alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(param)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaIL on a LIBERO suite")
    parser.add_argument("--suite", required=True, choices=list(SUITE_TO_DATASET_DIR))
    parser.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="Demos per task (default: all 50 = 100%% of data, matching OpenVLA/OFT "
        "finetuning; MaIL's paper default is 10 = 20%%).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast")
    parser.add_argument(
        "--data-dir", default=None, help="Default: LIBERO/libero/datasets/<task_suite_name>"
    )
    parser.add_argument("--output-dir", default="trained")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    _ensure_mail_importable()
    from agents.utils import ActionScaler
    from dataset.multi_task_dataset_aug import MultiTaskDataset

    dataset_dir_name = SUITE_TO_DATASET_DIR[args.suite]
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path("LIBERO/libero/datasets") / dataset_dir_name
    )
    arch = SUITE_ARCH_CONFIG[args.suite]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not any(data_dir.glob("*.hdf5")):
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    print(f"[{args.suite}] loading dataset from {data_dir}...")
    dataset = MultiTaskDataset(
        data_directory=str(data_dir),
        obs_keys="rgb",
        obs_modalities=OBS_MODALITIES,
        dataset_keys=["actions", "dones", "obs", "rewards", "states"],
        filter_by_attribute=None,
        padding=False,
        device="cpu",
        obs_dim=64,
        action_dim=arch.action_dim,
        state_dim=110,
        max_len_data=MAX_LEN_DATA,
        window_size=arch.window_size,
        num_data=args.num_demos if args.num_demos is not None else 50,
        data_aug=False,
    )
    print(f"[{args.suite}] {len(dataset)} training windows (num_demos={args.num_demos or 'all'})")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    scaler = ActionScaler(dataset.get_all_actions(), scale_data=True, device=device)

    policy = MailPolicy(arch, num_cameras=2, device=device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(f"[{args.suite}] policy: {num_params / 1e6:.2f}M params")
    policy.set_action_bounds(scaler.y_bounds_tensor[0], scaler.y_bounds_tensor[1])

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=0)
    ema = EmaTracker(policy, decay=args.ema_decay)

    run_dir = Path(args.output_dir) / f"mail_{args.suite}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "mean_loss", "epoch_time_s"])

    print(
        f"\n\n=== Training mail / {args.suite} ({device}) -- {num_params / 1e6:.2f}M params ===\n"
        f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  "
        f"ema_decay={args.ema_decay}  amp={args.amp}\n"
        f"num_demos={args.num_demos or 'all (50)'}  num_workers={args.num_workers}  seed={args.seed}\n"
        f"data_dir={data_dir}\n" 
        f"output_dir={run_dir}\n"
    )

    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_t0 = time.perf_counter()
        policy.train()
        losses = []
        batch_bar = tqdm(
            loader, desc=f"[{args.suite}] epoch {epoch + 1}/{args.epochs}", unit="batch"
        )
        for agentview, wrist, actions, mask in batch_bar:
            agentview = agentview.to(device, non_blocking=True)
            wrist = wrist.to(device, non_blocking=True)
            actions = scaler.scale_output(actions.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss = policy.loss(agentview, wrist, actions, mask=mask)
            loss.backward()
            optimizer.step()
            ema.update(policy)
            losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = sum(losses) / len(losses)
        epoch_time_s = time.perf_counter() - epoch_t0
        print(f"[{args.suite}] epoch {epoch + 1}/{args.epochs}  loss={mean_loss:.4f}  ({epoch_time_s:.1f}s)")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, mean_loss, epoch_time_s])

        meta = {
            "arch": arch.__dict__,
            "num_cameras": 2,
            "suite": args.suite,
            "num_demos": args.num_demos,
            "seed": args.seed,
            "epoch": epoch + 1,
        }
        save_checkpoint(run_dir / "last.pt", policy, scaler, ema.shadow, meta)
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_checkpoint(run_dir / "best.pt", policy, scaler, ema.shadow, meta)

    print(f"[{args.suite}] training done. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
