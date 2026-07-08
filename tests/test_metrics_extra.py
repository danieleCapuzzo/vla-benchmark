"""Additional edge-case coverage for core/metrics.py not covered by tests/test_metrics.py."""

import math

import numpy as np
import pytest

from core.metrics import aggregate_durations, aggregate_smoothness, compute_smoothness


def test_compute_smoothness_finite_on_libero_shaped_mixed_action():
    # (T, 7): 6 continuous dims + 1 near-binary gripper dim, as produced by the evaluator
    # before normalize_gripper_action/invert_gripper_action are applied.
    rng = np.random.default_rng(1)
    continuous = rng.normal(scale=0.01, size=(20, 6)).astype(np.float32)
    gripper = np.where(rng.random(20) > 0.5, 1.0, 0.0).astype(np.float32).reshape(-1, 1)
    actions = np.concatenate([continuous, gripper], axis=1)
    result = compute_smoothness(actions)
    assert math.isfinite(result)
    assert result >= 0.0


def test_compute_smoothness_constant_trajectory_is_zero():
    actions = np.ones((10, 7), dtype=np.float32) * 0.5
    assert compute_smoothness(actions) == pytest.approx(0.0)


def test_aggregate_durations_nonempty_episodes_all_failed():
    # Uncovered branch: `successes` is empty but `episodes` is not -- success-only means
    # fall back to 0.0 via the ternary (aggregate_durations, metrics.py:73,77), distinct
    # from the all-episode means which use the real data.
    episodes = [
        {"success": False, "num_steps": 220, "duration_s": 5.5},
        {"success": False, "num_steps": 180, "duration_s": 4.0},
    ]
    result = aggregate_durations(episodes)
    assert result["mean_steps_success"] == 0.0
    assert result["mean_duration_success_s"] == 0.0
    assert result["mean_steps_all"] == pytest.approx(200.0)
    assert result["mean_duration_all_s"] == pytest.approx(4.75)


def test_aggregate_smoothness_success_subset_ignores_nan_but_keeps_other_success():
    episodes = [
        {"success": True, "smoothness": float("nan")},
        {"success": True, "smoothness": 4.0},
        {"success": False, "smoothness": 1.0},
    ]
    result = aggregate_smoothness(episodes)
    assert result["mean_succ_smoothness"] == pytest.approx(4.0)
    assert result["mean_smoothness"] == pytest.approx(2.5)  # nanmean([nan, 4.0, 1.0])
