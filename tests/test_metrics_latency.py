"""Unit tests for core/metrics.aggregate_latency: unit conversions, throughput, percentiles."""

import math

import pytest

from core.metrics import aggregate_latency


def test_aggregate_latency_empty_returns_zeros():
    result = aggregate_latency([], chunk_size=8)
    assert result == {
        "lat_mean_ms": 0.0,
        "lat_median_ms": 0.0,
        "lat_std_ms": 0.0,
        "lat_var_ms2": 0.0,
        "lat_p95_ms": 0.0,
        "lat_p99_ms": 0.0,
        "effective_aps": 0.0,
        "feedback_hz": 0.0,
    }


def test_aggregate_latency_known_sample_values():
    latencies_s = [0.1, 0.2, 0.3]
    result = aggregate_latency(latencies_s, chunk_size=8)

    assert result["lat_mean_ms"] == pytest.approx(200.0)
    assert result["lat_median_ms"] == pytest.approx(200.0)
    assert result["lat_std_ms"] == pytest.approx(81.6496580927726)
    assert result["lat_var_ms2"] == pytest.approx(6666.666666666667)


def test_aggregate_latency_var_ms2_equals_std_ms_squared():
    # lat_var_ms2 (x1e6) and lat_std_ms (x1e3) are scaled independently in source
    # (metrics.py:36-37) -- pin the invariant that they still agree numerically.
    latencies_s = [0.05, 0.11, 0.09, 0.30, 0.12]
    result = aggregate_latency(latencies_s, chunk_size=1)
    assert result["lat_var_ms2"] == pytest.approx(result["lat_std_ms"] ** 2)


def test_aggregate_latency_throughput_derived_from_mean_not_tail():
    # effective_aps/feedback_hz track lat_mean_ms only -- a single large outlier moves
    # p99 a lot but barely moves throughput. Documents the "no warm-up discard" gap:
    # a cold first call is not excluded anywhere in the pipeline.
    latencies_s = [0.1] * 19 + [5.0]
    result = aggregate_latency(latencies_s, chunk_size=8)

    lat_mean_s = result["lat_mean_ms"] / 1e3
    assert result["effective_aps"] == pytest.approx(8 / lat_mean_s)
    assert result["feedback_hz"] == pytest.approx(1.0 / lat_mean_s)
    # the outlier dominates p99 but effective_aps stays close to the steady-state value
    assert result["lat_p99_ms"] > result["lat_mean_ms"]


def test_aggregate_latency_effective_aps_equals_chunk_size_times_feedback_hz():
    latencies_s = [0.05, 0.08, 0.12, 0.07]
    result = aggregate_latency(latencies_s, chunk_size=8)
    assert result["effective_aps"] == pytest.approx(8 * result["feedback_hz"])


def test_aggregate_latency_chunk_size_one_closed_loop_identity():
    latencies_s = [0.05, 0.08, 0.12, 0.07]
    result = aggregate_latency(latencies_s, chunk_size=1)
    assert result["effective_aps"] == pytest.approx(result["feedback_hz"])


def test_aggregate_latency_percentile_ordering():
    latencies_s = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.20, 0.50]
    result = aggregate_latency(latencies_s, chunk_size=8)
    assert result["lat_p99_ms"] >= result["lat_p95_ms"] >= result["lat_median_ms"]


def test_aggregate_latency_single_element_zero_spread():
    result = aggregate_latency([0.1], chunk_size=8)
    assert result["lat_std_ms"] == pytest.approx(0.0)
    assert result["lat_var_ms2"] == pytest.approx(0.0)
    assert result["lat_p95_ms"] == pytest.approx(100.0)
    assert result["lat_p99_ms"] == pytest.approx(100.0)
    assert math.isfinite(result["effective_aps"])


def test_aggregate_latency_ignores_cache_hit_zeros():
    # wrappers that internally cache an action chunk (e.g. MaIL) report raw_latency_s=0.0
    # on steps that reuse the cache instead of calling the model -- those must not drag
    # lat_median_ms/lat_mean_ms toward zero.
    latencies_s = [0.1] + [0.0] * 7
    result = aggregate_latency(latencies_s, chunk_size=1)
    assert result["lat_mean_ms"] == pytest.approx(100.0)
    assert result["lat_median_ms"] == pytest.approx(100.0)


def test_aggregate_latency_all_zeros_returns_empty_result():
    result = aggregate_latency([0.0, 0.0, 0.0], chunk_size=1)
    assert result["lat_mean_ms"] == 0.0
    assert result["lat_median_ms"] == 0.0
    assert result["effective_aps"] == 0.0
