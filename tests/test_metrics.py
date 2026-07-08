"""Unit tests for core/metrics.py: smoothness NaN guard, duration/smoothness aggregation."""

import math

import numpy as np
import pytest

from core.metrics import (
    compute_smoothness,
    aggregate_durations,
    aggregate_smoothness,
)

from results_io.results import nan_to_none


@pytest.mark.parametrize("num_steps", [0, 1, 2, 3])
def test_compute_smoothness_nan_guard_below_four_actions(num_steps):
    actions = np.zeros((num_steps, 7), dtype=np.float32)
    assert math.isnan(compute_smoothness(actions))


@pytest.mark.parametrize("num_steps", [4, 10])
def test_compute_smoothness_returns_finite_for_four_or_more(num_steps):
    rng = np.random.default_rng(0)
    actions = rng.normal(size=(num_steps, 7)).astype(np.float32)
    result = compute_smoothness(actions)
    assert math.isfinite(result)


def test_compute_smoothness_zero_for_constant_velocity():
    # A perfectly linear ramp has zero third discrete difference (jerk == 0.0).
    t = np.arange(10, dtype=np.float32).reshape(-1, 1)
    actions = np.tile(t, (1, 7))
    assert compute_smoothness(actions) == pytest.approx(0.0)


def test_nan_to_none_maps_nan_to_none_and_passes_through_floats():
    assert nan_to_none(float("nan")) is None
    assert nan_to_none(1.5) == 1.5


def test_aggregate_smoothness_nanmean_ignores_nan_episodes():
    episodes = [
        {"success": True, "smoothness": 1.0},
        {"success": True, "smoothness": float("nan")},
        {"success": False, "smoothness": 3.0},
    ]
    result = aggregate_smoothness(episodes)
    assert result["mean_smoothness"] == pytest.approx(2.0)  # nanmean([1.0, nan, 3.0])
    assert result["mean_succ_smoothness"] == pytest.approx(
        1.0
    )  # only the success, non-nan


def test_aggregate_smoothness_empty_episodes_returns_zero():
    result = aggregate_smoothness([])
    assert result == {"mean_smoothness": 0.0, "mean_succ_smoothness": 0.0}


def test_aggregate_smoothness_all_nan_returns_nan():
    episodes = [{"success": True, "smoothness": float("nan")}]
    result = aggregate_smoothness(episodes)
    assert math.isnan(result["mean_smoothness"])
    assert math.isnan(result["mean_succ_smoothness"])


def test_aggregate_durations_success_vs_all():
    episodes = [
        {"success": True, "num_steps": 10, "duration_s": 1.0},
        {"success": True, "num_steps": 20, "duration_s": 2.0},
        {"success": False, "num_steps": 100, "duration_s": 5.0},
    ]
    result = aggregate_durations(episodes)
    assert result["mean_steps_success"] == pytest.approx(15.0)
    assert result["mean_steps_all"] == pytest.approx(130.0 / 3)
    assert result["mean_duration_success_s"] == pytest.approx(1.5)
    assert result["mean_duration_all_s"] == pytest.approx(8.0 / 3)


def test_aggregate_durations_empty_episodes_returns_zero():
    result = aggregate_durations([])
    assert result == {
        "mean_steps_success": 0.0,
        "mean_steps_all": 0.0,
        "mean_duration_success_s": 0.0,
        "mean_duration_all_s": 0.0,
    }
