"""
Metric calculation utilities file.
Every function consumes the processed `PolicyOutput.actions` shape `(K, action_dim)`
or plain per-episode dicts, making it model-agnostic and easily extensible.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


def aggregate_latency(
    all_latencies_s: List[float], chunk_size: int
) -> Dict[str, float]:
    """Aggregates per-call model latencies into mean/median/std/var/p95/p99 (ms) plus the
    architecture-dependent throughput pair (see doc/ARCHITECTURE.md 5.2): effective actions/s
    scales by `chunk_size` (K=1 closed-loop, K>1 chunked), feedback_hz never does.
    """
    if not all_latencies_s:
        return {
            "lat_mean_ms": 0.0,
            "lat_median_ms": 0.0,
            "lat_std_ms": 0.0,
            "lat_var_ms2": 0.0,
            "lat_p95_ms": 0.0,
            "lat_p99_ms": 0.0,
            "effective_aps": 0.0,
            "feedback_hz": 0.0,
        }

    lat_mean_ms = float(np.mean(all_latencies_s) * 1e3)
    lat_median_ms = float(np.median(all_latencies_s) * 1e3)
    lat_std_ms = float(np.std(all_latencies_s) * 1e3)
    lat_var_ms2 = float(np.var(all_latencies_s) * 1e6)
    lat_p95_ms = float(np.percentile(all_latencies_s, 95) * 1e3)
    lat_p99_ms = float(np.percentile(all_latencies_s, 99) * 1e3)

    lat_mean_s = lat_mean_ms / 1e3
    effective_aps = (chunk_size / lat_mean_s) if lat_mean_s > 0 else 0.0
    feedback_hz = (1.0 / lat_mean_s) if lat_mean_s > 0 else 0.0

    return {
        "lat_mean_ms": lat_mean_ms,
        "lat_median_ms": lat_median_ms,
        "lat_std_ms": lat_std_ms,
        "lat_var_ms2": lat_var_ms2,
        "lat_p95_ms": lat_p95_ms,
        "lat_p99_ms": lat_p99_ms,
        "effective_aps": effective_aps,
        "feedback_hz": feedback_hz,
    }


def aggregate_durations(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates per-episode step counts and wall-clock durations (`duration_s`), split into
    success-only and all-episodes variants.
    """
    if not episodes:
        return {
            "mean_steps_success": 0.0,
            "mean_steps_all": 0.0,
            "mean_duration_success_s": 0.0,
            "mean_duration_all_s": 0.0,
        }

    successes = [ep for ep in episodes if ep["success"]]

    return {
        "mean_steps_success": (
            float(np.mean([ep["num_steps"] for ep in successes])) if successes else 0.0
        ),
        "mean_steps_all": float(np.mean([ep["num_steps"] for ep in episodes])),
        "mean_duration_success_s": (
            float(np.mean([ep["duration_s"] for ep in successes])) if successes else 0.0
        ),
        "mean_duration_all_s": float(np.mean([ep["duration_s"] for ep in episodes])),
    }


def compute_smoothness(actions: np.ndarray) -> float:
    """Mean action jerk: mean absolute third discrete difference of a stacked per-step action
    trajectory `(T, action_dim)`. Returns NaN when T < 4 (a third difference needs at least
    4 points). Format-agnostic -- works for any action_dim, closed-loop or chunked, as long as
    the caller has already flattened the episode into one row per control step.
    """
    if actions.shape[0] < 4:
        return float("nan")
    jerk = np.diff(actions, n=3, axis=0)
    return float(np.mean(np.abs(jerk)))


def aggregate_smoothness(episodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates per-episode `smoothness` (mean action jerk) across all episodes and across
    successes only. NaN-valued episodes (too few steps) are excluded via nanmean rather than
    corrupting the whole run's average.
    """
    if not episodes:
        return {"mean_smoothness": 0.0, "mean_succ_smoothness": 0.0}

    successes = [ep for ep in episodes if ep["success"]]
    all_values = [ep["smoothness"] for ep in episodes]
    succ_values = [ep["smoothness"] for ep in successes]

    def _nanmean_or_nan(values: List[float]) -> float:
        if not values or all(math.isnan(v) for v in values):
            return float("nan")
        return float(np.nanmean(values))

    return {
        "mean_smoothness": _nanmean_or_nan(all_values),
        "mean_succ_smoothness": _nanmean_or_nan(succ_values),
    }
