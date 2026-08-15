#!/usr/bin/env python
"""
Training script and CLI entrypoint for RoboMamba (frozen CLIP+Mamba trunk, trained head
only) on a single libero suite passed as a CLI argument.

Run in the `bench-env` conda env:
    python train_robomamba.py --suite spatial
    python train_robomamba.py --suite long

Two stages with separate (frozen!) trunk feature extraction:
  1. Extract: one forward pass of the frozen trunk over all demos/task, cache
     (features fp16 (N,2560), actions fp32 (N,ACTION_CHUNK,7) raw per-frame chunks, demo_ids
     int64 (N,)) to trained/robomamba/<suite>/features.pt.
     Demos are split into train/val (--val-frac default 0.1 val)
     Skipped if the cache already exists.

  2. Train: AdamW + L1 loss on all 7 action dims, on the cached features.
     Targets are per-dim bounds-normalized to [-1, 1] via q01/q99 (see core.core_utils).
     Trunk is not touched again -- this stage is seconds per epoch. Trains for a fixed
     number of epochs (--epochs); no validation split or early stopping.

Architecture: see core/arch/robomamba_arch.py. Data loading: see core/dataproc/libero_dataset.py.
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
from core.core_utils import normalize_actions
from core.arch.robomamba_arch import (
    ACTION_CHUNK,
    ACTION_DIM,
    IMAGE_SIZE,
    PROMPT_TEMPLATE,
    RoboMambaHead,
    build_trunk,
    save_checkpoint,
)
from core.dataproc.libero_dataset import extract_features


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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--trunk-checkpoint",
        default="trained/robomamba/RoboMamba-224-llava-R300-checkpoint.pth",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Dataset folder to train on. Default: datasets/<task_suite_name>_regen. "
        "Raises RuntimeError if the resolved directory does not exist.",
    )
    parser.add_argument("--output-dir", default="trained")
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-run stage 1 (feature extraction) even if a cache already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train only on the given dataset folder, defaulting to the regenerated dataset
    dataset_dir_name = SUITE_TO_DATASET_DIR[args.suite]
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = LIBERO_DATASETS / f"{dataset_dir_name}_regen"
    if not data_dir.exists():
        raise RuntimeError(f"Dataset directory not found: {data_dir}")
    output_dir = Path(args.output_dir)
    run_dir = output_dir / "robomamba" / args.suite

    # === STAGE 1: feature extraction ===
    # build the cache path and checkl if exists
    cache_path = output_dir / "robomamba" / args.suite / "features.pt"
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
    unique_demos = np.unique(demo_ids.numpy())

    # action bounds normalization (q01/q99, same scheme as OpenVLA-OFT's BOUNDS_Q99);
    # actions[:, 0] is each frame's own (unclamped) raw action -- equivalent to the pre-chunking
    # flat (N,7) cache this used to be computed on.
    action_q01 = torch.quantile(actions[:, 0], 0.01, dim=0)
    action_q99 = torch.quantile(actions[:, 0], 0.99, dim=0)
    print(
        f"[{args.suite}] action normalization: ON (bounds_q99)\n"
        f"  q01 {action_q01.cpu().numpy().round(4)}\n  q99  {action_q99.cpu().numpy().round(4)}"
    )

    targets = normalize_actions(actions, action_q01, action_q99)
    print(f"[{args.suite}] chunk={ACTION_CHUNK} actions  head input {features.shape[1]}-d")

    # build torch tensor dataset and dataloader
    train_dataset = TensorDataset(features, targets)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # create the head
    head = RoboMambaHead().to(device)
    num_params = sum(p.numel() for p in head.parameters())
    print(
        f"[{args.suite}] head: {num_params / 1e6:.2f}M params  "
        f"train {len(train_dataset)} frames ({len(unique_demos)} demos)"
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.L1Loss()

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "epoch_time_s"])

    print(
        f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  "
        f"weight_decay={args.weight_decay}  seed={args.seed}\n"
        f"data_dir={data_dir}\noutput_dir={run_dir}\n"
    )

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

        epoch_time_s = time.perf_counter() - epoch_t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, epoch_time_s])
        epoch_bar.set_postfix(train=f"{train_loss:.4f}")

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
            "action_chunk": ACTION_CHUNK,
            "action_q01": action_q01.cpu().numpy(),
            "action_q99": action_q99.cpu().numpy(),
            "train_loss": train_loss,
        }
        save_checkpoint(run_dir / "last.pt", head, meta)

    print(f"[{args.suite}] training done. final train_loss={train_loss:.4f}. checkpoint in {run_dir}")


if __name__ == "__main__":
    main()
