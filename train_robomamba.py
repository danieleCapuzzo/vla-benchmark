#!/usr/bin/env python
"""
Training script and CLI entrypoint for RoboMamba (frozen CLIP+Mamba trunk, trained head
only) on a single libero suite passed as a CLI argument.

Run in the `bench-env` conda env:
    python train_robomamba.py --suite spatial
    python train_robomamba.py --suite long

Two stages with separate (frozen!) trunk feature extraction:
  1. Extract: one forward pass of the frozen trunk over all demos/task, cache
     (features fp16 (N,2560), actions fp32 (N,7), demo_ids int64 (N,)) to
     trained/robomamba/<suite>/features.pt. 
     Demos are split into train/val (--val-frac default 0.1 val)
     Skipped if the cache already exists.

  2. Train: AdamW + L1 loss on all 7 action dims, on the cached features.
     Targets are per-dim bounds-normalized to [-1, 1] via q01/q99 (see core.core_utils),
     val_loss reported in raw action units.
     Trunk is not touched again -- this stage is seconds per epoch. Demos are split train/val (--val-frac,
     Uses early stopping on val-loss (--patience defaults to 10).

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

from config.constants import NUM_CAMERAS_BY_MODEL, SUITE_TO_DATASET_DIR
from config.paths import LIBERO_DATASETS
from core.core_utils import denormalize_actions, normalize_actions
from core.robomamba_arch import (
    ACTION_CHUNK,
    ACTION_DIM,
    HISTORY_FRAMES,
    IMAGE_SIZE,
    PROMPT_TEMPLATE,
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
        default=128,
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
        "--data-dir", default=None,
        help="Default: datasets/<task_suite_name>_regen, falling back to "
        "datasets/<task_suite_name> if the regenerated dataset is not found.",
    )
    parser.add_argument("--output-dir", default="trained")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run stage 1 (feature extraction) even if a cache already exists.",
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

    # train on regen data if possible
    dataset_dir_name = SUITE_TO_DATASET_DIR[args.suite]
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        regen_dir = LIBERO_DATASETS / f"{dataset_dir_name}_regen"
        if regen_dir.exists():
            data_dir = regen_dir
        else:
            data_dir = LIBERO_DATASETS / dataset_dir_name
            print(f"Regenerated data not found, using {data_dir}")
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "robomamba" / args.suite

    # === STAGE 1: feature extraction ===
    # build the cache path and checkl if exists
    cache_path = feature_cache_path(output_dir, args.suite)
    cache_hit = cache_path.exists() and not args.force_extract
    print(f"\n\n=== STAGE 1: feature extraction ({args.suite}) ===")
    print(f"cache path: {cache_path}")
    print(f"cache {'HIT' if cache_hit else 'MISS'} -- {'reusing cached features' if cache_hit else 'will extract from the frozen trunk'}")

    # reuse features if in cache
    if cache_hit:
        cached = torch.load(cache_path, map_location="cpu")
        features, actions, demo_ids = cached["features"], cached["actions"], cached["demo_ids"]
        print(f"[{args.suite}] loaded {features.shape[0]} cached frames")
    else:
        # build robomamba trunk
        print(f"[{args.suite}] loading RoboMamba trunk from {args.trunk_checkpoint}...")
        t0 = time.perf_counter()
        trunk = build_trunk(Path(args.trunk_checkpoint), device=device)
        print(f"[{args.suite}] trunk loaded in {time.perf_counter() - t0:.1f}s")

        # extract features
        print(f"[{args.suite}] extracting features from {data_dir}...")
        t0 = time.perf_counter()
        features, actions, demo_ids = extract_features(
            trunk,
            data_dir=data_dir,
            suite=args.suite,
            batch_size=args.extract_batch_size,
            device=device,
        )
        print(
            f"[{args.suite}] extracted {features.shape[0]} frames in "
            f"{time.perf_counter() - t0:.1f}s"
        )

        # save the features as tensors
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": features, "actions": actions, "demo_ids": demo_ids}, cache_path)
        print(f"[{args.suite}] cached features to {cache_path}")

        del trunk
        gc.collect()
        torch.cuda.empty_cache()

    # === STAGE 2: train head on cached features ===
    print(f"\n\n=== STAGE 2: head training ({args.suite}) ===")

    features = features.to(device=device, dtype=torch.float32)
    actions = actions.to(device=device, dtype=torch.float32)

    # train/valsplit by demo (not frame)
    rng = np.random.RandomState(args.seed)
    unique_demos = np.unique(demo_ids.numpy())
    rng.shuffle(unique_demos)
    n_val_demos = max(1, int(len(unique_demos) * args.val_frac))
    val_demos = set(unique_demos[:n_val_demos].tolist())
    is_val = torch.tensor([d.item() in val_demos for d in demo_ids])

    # action bounds normalization (q01/q99, same scheme as OpenVLA-OFT's BOUNDS_Q99) in TRAIN ONLY
    action_q01 = torch.quantile(actions[~is_val], 0.01, dim=0)
    action_q99 = torch.quantile(actions[~is_val], 0.99, dim=0)
    print(
        f"[{args.suite}] action normalization: ON (bounds_q99)\n"
        f"  q01 {action_q01.cpu().numpy().round(4)}\n  q99  {action_q99.cpu().numpy().round(4)}"
    )

    # [f(t-K), f(t)] -> next K actions
    features, chunked_actions, seq_demo_ids = build_history_chunks(features, actions, demo_ids)
    is_val = torch.tensor([d.item() in val_demos for d in seq_demo_ids])
    targets = normalize_actions(chunked_actions, action_q01, action_q99)
    print(
        f"[{args.suite}] history={HISTORY_FRAMES} frames (gap {ACTION_CHUNK})  "
        f"chunk={ACTION_CHUNK} actions  head input {features.shape[1]}-d"
    )

    # build torch tensor datasets and dataloaders
    train_dataset = TensorDataset(features[~is_val], targets[~is_val])
    val_dataset = TensorDataset(features[is_val], targets[is_val])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # create the head
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
                # de-normalize both sides so val_loss is reported in raw LIBERO action units
                pred = denormalize_actions(head(batch_features), action_q01, action_q99)
                target = denormalize_actions(batch_targets, action_q01, action_q99)
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
            "data_dir": str(data_dir),
            "image_size": IMAGE_SIZE,
            "prompt_template": PROMPT_TEMPLATE,
            "action_dim": ACTION_DIM,
            "num_cameras": NUM_CAMERAS_BY_MODEL["robomamba"],
            "upright_images": True,
            "readout": "language_pool",
            "history_frames": HISTORY_FRAMES,
            "action_chunk": ACTION_CHUNK,
            "action_q01": action_q01.cpu().numpy(),
            "action_q99": action_q99.cpu().numpy(),
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
