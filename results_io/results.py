"""
Serialization of per-episode and aggregated run results.

Output format:
    per doc/ARCHITECTURE.md: episodes.jsonl (one line per episode)
    and summary.csv (one aggregated row per suite).

Note:
    if more than one suite is executed, an overall summary file
    named summary_<suite1>-<suite2>-<suiteN> will be produced
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.policy_wrapper import PolicyMetadata

SUMMARY_CSV_FIELDS = [
    "model_id",
    "suite",
    "quant",  # quantization
    "n_cam",  # number of cameras
    "seed",
    "success_rate",
    "n_episodes",
    "lat_mean_ms",
    "lat_median_ms",
    "lat_std_ms",
    "lat_var_ms2",
    "lat_p95_ms",  # 99th percentile latency: at least 99% of model calls were this fast
    "lat_p99_ms",  # 95th percentile latency
    "effective_aps",
    "feedback_hz",
    "vram_peak_gb",  # max consumed VRAM
    "num_params",
    "mean_steps_success",  # mean number of steps for success
]


def write_episodes_jsonl(path: str | Path, episodes: List[Dict[str, Any]]) -> None:
    """Writes one JSON object per line, one per episode.

    Args:
        path: Destination file path (typically ".../episodes.jsonl").
        episodes: List of per-episode result dicts as produced by
            `LiberoEvaluator._run_episode` (keys: suite, task_id, trial, success, num_steps,
            latencies_s, lat_mean_s, is_chunked, chunk_size).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")


def write_summary_csv(
    path: str | Path,
    episodes: List[Dict[str, Any]],
    metadata: PolicyMetadata,
    seed: int,
    vram_peak_gb: float,
) -> Dict[str, Any]:
    """Writes a single aggregated summary row for this run.

    Args:
        path: Destination file path (typically ".../summary.csv").
        episodes: All episode result dicts for this run (see `write_episodes_jsonl`).
        metadata: The model's static metadata (from `wrapper.get_metadata()`).
        seed: Random seed used for this run.
        vram_peak_gb: Peak VRAM in GB observed during this run.

    Returns:
        The row dict that was written, keyed by `SUMMARY_CSV_FIELDS` -- reused by callers
        that aggregate multiple suite runs into a cross-run summary (see
        `write_cross_run_summary`) without re-reading the file back.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_episodes = len(episodes)
    successes = [ep for ep in episodes if ep["success"]]
    success_rate = len(successes) / n_episodes if n_episodes else 0.0

    all_latencies_s = [lat for ep in episodes for lat in ep["latencies_s"]]
    lat_mean_ms = float(np.mean(all_latencies_s) * 1e3) if all_latencies_s else 0.0
    lat_median_ms = float(np.median(all_latencies_s) * 1e3) if all_latencies_s else 0.0

    # std/var of latency in ms
    # large values --> inconsistent real-time behavior.
    lat_std_ms = float(np.std(all_latencies_s) * 1e3) if all_latencies_s else 0.0
    lat_var_ms2 = float(np.var(all_latencies_s) * 1e6) if all_latencies_s else 0.0

    # 95/99 percentile latencies
    # very informative metrics for real time
    lat_p95_ms = (
        float(np.percentile(all_latencies_s, 95) * 1e3) if all_latencies_s else 0.0
    )
    lat_p99_ms = (
        float(np.percentile(all_latencies_s, 99) * 1e3) if all_latencies_s else 0.0
    )

    lat_mean_s = lat_mean_ms / 1e3
    chunk_size = episodes[0]["chunk_size"] if episodes else 1
    effective_aps = (chunk_size / lat_mean_s) if lat_mean_s > 0 else 0.0
    feedback_hz = (1.0 / lat_mean_s) if lat_mean_s > 0 else 0.0

    mean_steps_success = (
        float(np.mean([ep["num_steps"] for ep in successes])) if successes else 0.0
    )

    row = {
        "model_id": metadata.model_id,
        "suite": episodes[0]["suite"] if episodes else "",
        "quant": metadata.quant,
        "n_cam": metadata.num_cameras,
        "seed": seed,
        "success_rate": success_rate,
        "n_episodes": n_episodes,
        "lat_mean_ms": lat_mean_ms,
        "lat_median_ms": lat_median_ms,
        "lat_std_ms": lat_std_ms,
        "lat_var_ms2": lat_var_ms2,
        "lat_p95_ms": lat_p95_ms,
        "lat_p99_ms": lat_p99_ms,
        "effective_aps": effective_aps,
        "feedback_hz": feedback_hz,
        "vram_peak_gb": vram_peak_gb,
        "num_params": metadata.num_params,
        "mean_steps_success": mean_steps_success,
    }

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    return row


def write_cross_run_summary(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """
    Writes a combined summary with one row per suite, for a multi-suite invocation.
    Vertical stack of per-suite summaries sharing the same `SUMMARY_CSV_FIELDS`
    --> loadable with pandas for cross-suite comparison tables.

    Args:
        path: destination file path (typically ".../suite_<s1>-<s2>-..._<timestamp>.csv").
        rows: one summary row dict per suite, as returned by `write_summary_csv`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
