#!/usr/bin/env python
"""
Training script and CLI entrypoint for RoboMamba (frozen CLIP+Mamba trunk, trained head
only) on a single libero suite passed as a CLI argument.

Run in the `bench-env` conda env:
    python train_robomamba.py --suite spatial
    python train_robomamba.py --suite long

Two stages, since the trunk is fully frozen and there's no image augmentation -- every
frame's 2560-d feature is a constant, so it's cheaper to compute it once:
  1. Extract: one forward pass of the frozen trunk over all 50 demos/task, cache
     (features fp16 (N,2560), actions fp32 (N,7), demo_ids int64 (N,)) to
     trained/robomamba/<suite>/features_<version>.pt. Skipped if the cache already
     exists (see --force-extract). The filename carries FEATURE_CACHE_VERSION, so changing
     the preprocessing or readout misses the old cache instead of silently reusing it.
  2. Train: AdamW + L1 loss on all 7 action dims, on the cached features. Targets are
     per-dim standardized (see --raw-actions) so the small rotation dims are not starved by
     xyz/gripper; val_loss is always reported in raw action units. Trunk is not touched
     again -- this stage is seconds per epoch. Demos are split train/val (--val-frac,
     default 0.1, split by demo so temporally-correlated frames from the same demo never
     straddle the split) and training stops early on val-loss plateau (--patience).

CAVEAT on model selection: val L1 is a weak proxy for closed-loop success. In a measured
task-1 ablation, val L1 moved only 0.186 -> 0.121 while success rate moved 0/10 -> 8/10.
Treat `best.pt` as a starting point and confirm with actual rollouts (run_eval.py).

Architecture and data loading: see core/robomamba_arch.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=r"`resume_download` is deprecated.*"
)

from core.core_utils import denormalize_actions, normalize_actions
from core.robomamba_arch import (
    ACTION_CHUNK,
    HISTORY_FRAMES,
    SUITE_TO_DATASET_DIR,
    RoboMambaHead,
    build_history_chunks,
    build_trunk,
    extract_features,
    feature_cache_path,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoboMamba's action head on a LIBERO suite")
    parser.add_argument("--suite", required=True, choices=list(SUITE_TO_DATASET_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256, help="Stage 2 (head-only) batch size.")
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=32,
        help="Stage 1 (trunk forward pass) batch size -- tune this to fit your GPU.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--trunk-checkpoint",
        default="trained/robomamba/RoboMamba-224-llava-R300-checkpoint.pth",
    )
    parser.add_argument(
        "--data-dir", default=None, help="Default: LIBERO/libero/datasets/<task_suite_name>"
    )
    parser.add_argument("--output-dir", default="trained")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run stage 1 (feature extraction) even if a cache already exists.",
    )
    parser.add_argument(
        "--raw-actions",
        action="store_true",
        help="Train on raw action targets instead of per-dim standardized ones. LIBERO's "
        "action dims differ in scale by >10x, so raw targets let xyz/gripper dominate the "
        "loss and leave the rotation dims effectively unlearned (see "
        "core.core_utils.normalize_actions). For ablation only.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of demos (not frames) held out for validation/early stopping.",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Stop if val loss hasn't improved for this many epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir_name = SUITE_TO_DATASET_DIR[args.suite]
    data_dir = (
        Path(args.data_dir) if args.data_dir else Path("LIBERO/libero/datasets") / dataset_dir_name
    )
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "robomamba" / args.suite

    # --- stage 1: extract (or reuse cached) trunk features ---
    cache_path = feature_cache_path(output_dir, args.suite)
    cache_hit = cache_path.exists() and not args.force_extract
    print(f"\n\n=== STAGE 1: feature extraction ({args.suite}) ===")
    print(f"cache path: {cache_path}")
    print(f"cache {'HIT' if cache_hit else 'MISS'} -- {'reusing cached features' if cache_hit else 'will extract from the frozen trunk'}")

    if cache_hit:
        cached = torch.load(cache_path, map_location="cpu")
        features, actions, demo_ids = cached["features"], cached["actions"], cached["demo_ids"]
        print(f"[{args.suite}] loaded {features.shape[0]} cached frames")
    else:
        print(f"[{args.suite}] loading RoboMamba trunk from {args.trunk_checkpoint}...")
        t0 = time.perf_counter()
        trunk = build_trunk(Path(args.trunk_checkpoint), device=device)
        print(f"[{args.suite}] trunk loaded in {time.perf_counter() - t0:.1f}s")

        print(f"[{args.suite}] extracting features from {data_dir}...")
        t0 = time.perf_counter()
        features, actions, demo_ids = extract_features(
            trunk,
            data_dir=data_dir,
            batch_size=args.extract_batch_size,
            device=device,
        )
        print(
            f"[{args.suite}] extracted {features.shape[0]} frames in "
            f"{time.perf_counter() - t0:.1f}s"
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "actions": actions, "demo_ids": demo_ids}, cache_path)
        print(f"[{args.suite}] cached features to {cache_path}")

        del trunk
        gc.collect()
        torch.cuda.empty_cache()

    # --- stage 2: train the head on cached features ---
    print(f"\n\n=== STAGE 2: head training ({args.suite}) ===")

    features = features.to(device=device, dtype=torch.float32)
    actions = actions.to(device=device, dtype=torch.float32)

    # split by demo (not frame) -- frames within one demo are highly temporally correlated,
    # a frame-level split would leak adjacent frames across train/val and hide overfitting.
    rng = np.random.RandomState(args.seed)
    unique_demos = np.unique(demo_ids.numpy())
    rng.shuffle(unique_demos)
    n_val_demos = max(1, int(len(unique_demos) * args.val_frac))
    val_demos = set(unique_demos[:n_val_demos].tolist())
    is_val = torch.tensor([d.item() in val_demos for d in demo_ids])

    # per-dim action standardization, computed on the TRAIN split only so no val statistics
    # leak into training. Stored in the checkpoint meta; the wrapper inverts it at eval.
    if args.raw_actions:
        action_mean = torch.zeros(actions.shape[1], device=device)
        action_std = torch.ones(actions.shape[1], device=device)
    else:
        action_mean = actions[~is_val].mean(dim=0)
        action_std = actions[~is_val].std(dim=0).clamp_min(1e-6)
    print(
        f"[{args.suite}] action normalization: {'OFF (--raw-actions)' if args.raw_actions else 'ON'}\n"
        f"  mean {action_mean.cpu().numpy().round(4)}\n  std  {action_std.cpu().numpy().round(4)}"
    )

    # [f(t-K), f(t)] -> next K actions. Rebuilds is_val, since this reorders by demo.
    features, chunked_actions, seq_demo_ids = build_history_chunks(features, actions, demo_ids)
    is_val = torch.tensor([d.item() in val_demos for d in seq_demo_ids])
    targets = normalize_actions(chunked_actions, action_mean, action_std)
    print(
        f"[{args.suite}] history={HISTORY_FRAMES} frames (gap {ACTION_CHUNK})  "
        f"chunk={ACTION_CHUNK} actions  head input {features.shape[1]}-d"
    )

    train_dataset = TensorDataset(features[~is_val], targets[~is_val])
    val_dataset = TensorDataset(features[is_val], targets[is_val])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    head = RoboMambaHead().to(device)
    num_params = sum(p.numel() for p in head.parameters())
    print(
        f"[{args.suite}] head: {num_params / 1e6:.2f}M params  "
        f"train {len(train_dataset)} frames ({len(unique_demos) - n_val_demos} demos)  "
        f"val {len(val_dataset)} frames ({n_val_demos} demos)"
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.L1Loss()

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "epoch_time_s"])

    print(
        f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  "
        f"weight_decay={args.weight_decay}  patience={args.patience}\n"
        f"val_frac={args.val_frac}  seed={args.seed}\n"
        f"data_dir={data_dir}\noutput_dir={run_dir}\n"
    )

    best_val_loss = float("inf")
    epochs_since_improvement = 0
    epoch_bar = tqdm(range(args.epochs), desc=f"[{args.suite}] training head", unit="epoch")
    for epoch in epoch_bar:
        epoch_t0 = time.perf_counter()
        head.train()
        train_losses = []
        for batch_features, batch_actions in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = head(batch_features)
            loss = loss_fn(pred, batch_actions)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)

        head.eval()
        val_losses = []
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                # de-normalize both sides so val_loss is always in raw LIBERO action units,
                # and therefore comparable across --raw-actions and normalized runs.
                pred = denormalize_actions(head(batch_features), action_mean, action_std)
                target = denormalize_actions(batch_targets, action_mean, action_std)
                val_losses.append(loss_fn(pred, target).item())
        val_loss = sum(val_losses) / len(val_losses)

        epoch_time_s = time.perf_counter() - epoch_t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, epoch_time_s])
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        meta = {
            "suite": args.suite,
            "seed": args.seed,
            "epoch": epoch + 1,
            "trunk_checkpoint": args.trunk_checkpoint,
            "image_size": 128,
            "prompt_template": "<|user|> <image>\n{instruction}\n<|assistant|>",
            "action_dim": 7,
            "num_cameras": 1,
            "upright_images": True,
            "readout": "language_pool",
            "history_frames": HISTORY_FRAMES,
            "action_chunk": ACTION_CHUNK,
            "action_mean": action_mean.cpu().numpy(),
            "action_std": action_std.cpu().numpy(),
            "val_loss": val_loss,
        }
        save_checkpoint(run_dir / "last.pt", head, meta)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            save_checkpoint(run_dir / "best.pt", head, meta)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                print(f"[{args.suite}] early stopping at epoch {epoch + 1} (no val improvement for {args.patience} epochs)")
                break

    print(f"[{args.suite}] training done. best val_loss={best_val_loss:.4f}. checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
