#!/usr/bin/env python
"""
CLI entrypoint: wires PolicyFactory -> wrapper -> LiberoEvaluator -> results_io.
Example:
    python run_eval.py --model oft --suite spatial --quant 8bit --num-cameras 1 --rollouts 20
    python run_eval.py --model oft --suite all --quant 8bit --num-cameras 2  # full benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

# import repo-root packages
# ensure clean import regardless of caller cwd
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from config.loader import FROZEN_ROLLOUTS_PER_TASK, RunConfig, load_config  # noqa: E402
from core.factory import PolicyFactory  # noqa: E402

# trigger wrapper registration using @PolicyFactory.register decorators.
# suppress warnings
import wrappers.openvla_wrapper  # noqa: E402,F401
import wrappers.oft_wrapper  # noqa: E402,F401
import wrappers.mail_wrapper  # noqa: E402,F401
from core.evaluator import LiberoEvaluator  # noqa: E402
from results_io.results import (  # noqa: E402
    write_cross_run_summary,
    write_episodes_jsonl,
    write_summary_csv,
)

ALL_SUITES = ["spatial", "object", "goal", "long"]


def parse_args() -> argparse.Namespace:
    """
    Parses keyboard arguments
    """
    parser = argparse.ArgumentParser(description="VLA-on-LIBERO testbench CLI")
    parser.add_argument("--model", required=True, choices=["openvla", "oft", "mail"])
    parser.add_argument(
        "--suite",
        required=True,
        nargs="+",
        choices=["spatial", "object", "goal", "long", "all"],
        help="One or more suites, or 'all' for the full benchmark in one invocation.",
    )
    parser.add_argument("--quant", default=None, choices=["none", "8bit", "4bit"])
    parser.add_argument("--num-cameras", type=int, default=None, choices=[1, 2])
    parser.add_argument(
        "--rollouts",
        type=int,
        default=None,
        help="Rollouts per task (protocol default: 20)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON (default: config/config.json)",
    )
    return parser.parse_args()


def resolve_suites(requested: List[str]) -> List[str]:
    """Expands 'all' and de-duplicates while preserving first-seen order."""
    if "all" in requested:
        return list(ALL_SUITES)
    seen = []
    for s in requested:
        if s not in seen:
            seen.append(s)
    return seen


def run_one_suite(
    *,
    model: str,
    suite: str,
    cfg: RunConfig,
    quant: str,
    num_cameras: int,
    rollouts: int,
    seed: int,
    output_dir_path: Path,
) -> Dict[str, Any]:
    """Runs a full evaluation for a single suite and writes its per-run result files.

    Args:
        model: Model id passed to `PolicyFactory.create` (e.g. "oft").
        suite: Suite name key into `cfg.suites` (e.g. "spatial").
        cfg: Loaded run configuration.
        quant: Quantization mode ("none", "8bit", "4bit").
        num_cameras: Number of cameras fed to the policy.
        rollouts: Rollouts per task for this run.
        seed: Random seed, recorded in the summary row.
        output_dir_path: Absolute base output directory (see `main`'s cwd-anchoring).

    Returns:
        The aggregated summary row dict for this suite (as written to its `summary.csv`),
        reused by `main` to assemble the cross-run summary when multiple suites are run.

    Raises:
        ValueError: If `suite` is not present in `cfg.suites`.
    """
    if suite not in cfg.suites:
        raise ValueError(
            f"Suite '{suite}' not found in config. Valid suites: {sorted(cfg.suites)}"
        )
    suite_cfg = cfg.suites[suite]

    # Fresh checkpoint per suite (each suite ships its own fine-tuned checkpoint) -- reset
    # peak-memory stats first so this suite's VRAM peak isn't inflated by the previous
    # suite's (now-freed) model.
    torch.cuda.reset_peak_memory_stats()

    wrapper = PolicyFactory.create(
        model,
        checkpoint=suite_cfg.checkpoint,
        unnorm_key=suite_cfg.unnorm_key,
    )
    wrapper.load(quant=quant, num_cameras=num_cameras)

    evaluator = LiberoEvaluator(
        task_suite_name=suite_cfg.task_suite_name,
        num_steps_wait=cfg.num_steps_wait,
    )
    episodes = evaluator.run(wrapper, num_rollouts_per_task=rollouts)

    metadata = wrapper.get_metadata()
    vram_peak_gb = wrapper.get_vram_footprint_gb()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir_path / f"{model}_{suite}_{quant}_{num_cameras}cam_{timestamp}"

    write_episodes_jsonl(run_dir / "episodes.jsonl", episodes)
    row = write_summary_csv(
        run_dir / "summary.csv",
        episodes,
        metadata,
        seed=seed,
        vram_peak_gb=vram_peak_gb,
    )

    print(f"[{suite}] Run complete. Results written to {run_dir}")
    print(f"[{suite}] VRAM peak: {vram_peak_gb:.2f} GB")

    # Free the model before the next suite loads its own checkpoint -- prevents VRAM
    # accumulation across suites on the 16 GB card and keeps each suite's peak independent.
    del wrapper
    torch.cuda.empty_cache()

    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    # default to configuration if not provided
    quant = args.quant if args.quant is not None else cfg.quant
    num_cameras = args.num_cameras if args.num_cameras is not None else cfg.num_cameras
    rollouts = args.rollouts if args.rollouts is not None else cfg.rollouts_per_task
    seed = args.seed if args.seed is not None else cfg.seed
    output_dir = args.output_dir if args.output_dir is not None else cfg.output_dir

    # create the output directory where we log the experiments
    # independent of caller cwd
    output_dir_path = Path(output_dir)
    if not output_dir_path.is_absolute():
        output_dir_path = REPO_ROOT / output_dir_path

    # warn of different num rollouts than expected
    if rollouts != FROZEN_ROLLOUTS_PER_TASK:
        print(
            f"WARNING: --rollouts {rollouts} deviates from the frozen protocol value "
            f"({FROZEN_ROLLOUTS_PER_TASK}/task, see doc/ARCHITECTURE.md). "
            "Use only for smoke tests, not for reported results."
        )

    suites = resolve_suites(args.suite)

    rows = [
        run_one_suite(
            model=args.model,
            suite=suite,
            cfg=cfg,
            quant=quant,
            num_cameras=num_cameras,
            rollouts=rollouts,
            seed=seed,
            output_dir_path=output_dir_path,
        )
        for suite in suites
    ]

    if len(suites) > 1:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cross_run_path = output_dir_path / f"suite_{'-'.join(suites)}_{timestamp}.csv"
        write_cross_run_summary(cross_run_path, rows)
        print(f"Cross-run summary ({len(suites)} suites) written to {cross_run_path}")


if __name__ == "__main__":
    main()
