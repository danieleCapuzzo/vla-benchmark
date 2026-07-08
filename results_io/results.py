"""
Serialization of per-episode and aggregated run results.

Output format:
    per doc/ARCHITECTURE.md: episodes.jsonl (one line per episode)
    and summary.csv (one aggregated row per suite).

Note:
    if more than one suite is executed, an overall summary file named
    all_suites.csv (one row per suite) will be produced at the run's top level.

All metric math (latency stats, durations, smoothness) is delegated to `core.metrics` --
this module only shapes the row/line and handles file I/O.
"""

from __future__ import annotations

import math
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from core.metrics import (
    aggregate_durations,
    aggregate_latency,
    aggregate_smoothness,
)
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
    "mean_steps_success",  # mean number of steps for successful episodes
    "mean_steps_all",  # mean number of steps across all episodes
    "mean_duration_success_s",  # mean wall-clock rollout duration for successful episodes
    "mean_duration_all_s",  # mean wall-clock rollout duration across all episodes
    "mean_smoothness",  # mean action jerk across all episodes
    "mean_succ_smoothness",  # mean action jerk across successful episodes only
]


def nan_to_none(value: float) -> float | None:
    """Maps NaN to None (valid JSON `null`); passes any other float through unchanged."""
    return None if math.isnan(value) else value


def write_episodes_jsonl(path: str | Path, episodes: List[Dict[str, Any]]) -> None:
    """Writes one JSON object per line, one per episode.

    Args:
        path: Destination file path (typically ".../episodes.jsonl").
        episodes: List of per-episode result dicts as produced by
            `LiberoEvaluator._run_episode` (keys: suite, task_id, trial, success, num_steps,
            latencies_s, lat_mean_s, is_chunked, chunk_size, duration_s, smoothness).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ep in episodes:
            ep = {**ep, "smoothness": nan_to_none(ep["smoothness"])}
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
        `write_all_suites_csv`) without re-reading the file back.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_episodes = len(episodes)
    successes = [ep for ep in episodes if ep["success"]]
    success_rate = len(successes) / n_episodes if n_episodes else 0.0

    all_latencies_s = [lat for ep in episodes for lat in ep["latencies_s"]]
    chunk_size = episodes[0]["chunk_size"] if episodes else 1

    row = {
        "model_id": metadata.model_id,
        "suite": episodes[0]["suite"] if episodes else "",
        "quant": metadata.quant,
        "n_cam": metadata.num_cameras,
        "seed": seed,
        "success_rate": success_rate,
        "n_episodes": n_episodes,
        **aggregate_latency(all_latencies_s, chunk_size),
        "vram_peak_gb": vram_peak_gb,
        "num_params": metadata.num_params,
        **aggregate_durations(episodes),
        **aggregate_smoothness(episodes),
    }
    # NaN has no valid CSV representation (the literal text "nan" isn't standard and trips
    # up strict readers); map to None so csv.DictWriter emits an empty cell instead, same
    # convention as write_episodes_jsonl's NaN -> JSON null.
    row = {k: (nan_to_none(v) if isinstance(v, float) else v) for k, v in row.items()}

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    return row


def write_all_suites_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """
    Writes a combined summary with one row per suite, for a multi-suite invocation.
    List of per-suite summary rows (same fields as `summary.csv`, see
    `SUMMARY_CSV_FIELDS`) --> loadable for cross-suite comparison.

    Args:
        path: destination file path (typically ".../all_suites.csv").
        rows: one summary row dict per suite, as returned by `write_summary_csv`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
