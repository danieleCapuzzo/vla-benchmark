"""Unit tests for results_io/results.py: episodes.jsonl, summary.csv, all_suites.csv."""

import csv
import json

import pytest

from core.policy_wrapper import PolicyMetadata
from results_io.results import (
    SUMMARY_CSV_FIELDS,
    write_all_suites_csv,
    write_episodes_jsonl,
    write_summary_csv,
)


def _fake_metadata(**overrides):
    defaults = dict(
        model_id="oft",
        variant="fake/checkpoint",
        num_params=7_541_237_184,
        quant="8bit",
        num_cameras=1,
        action_chunk_size=8,
        attn_impl="sdpa",
        extra={"unnorm_key": "libero_spatial_no_noops"},
    )
    defaults.update(overrides)
    return PolicyMetadata(**defaults)


def _episode(**overrides):
    defaults = dict(
        suite="libero_spatial",
        task_id=0,
        trial=0,
        success=True,
        num_steps=220,
        latencies_s=[0.1, 0.11, 0.12],
        lat_mean_s=0.11,
        is_chunked=True,
        chunk_size=8,
        duration_s=5.5,
        smoothness=0.15,
    )
    defaults.update(overrides)
    return defaults


def test_write_episodes_jsonl_one_line_per_episode_with_full_key_set(tmp_path):
    episodes = [_episode(trial=0), _episode(trial=1, success=False)]
    path = tmp_path / "episodes.jsonl"

    write_episodes_jsonl(path, episodes)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    loaded = [json.loads(line) for line in lines]
    assert loaded[0]["trial"] == 0
    assert loaded[1]["success"] is False
    for ep in loaded:
        assert set(ep.keys()) == set(episodes[0].keys())


def test_write_episodes_jsonl_nan_smoothness_becomes_json_null(tmp_path):
    episodes = [_episode(smoothness=float("nan"))]
    path = tmp_path / "episodes.jsonl"

    write_episodes_jsonl(path, episodes)

    loaded = json.loads(path.read_text().splitlines()[0])
    assert loaded["smoothness"] is None


def test_write_episodes_jsonl_finite_smoothness_passes_through(tmp_path):
    episodes = [_episode(smoothness=0.4242)]
    path = tmp_path / "episodes.jsonl"

    write_episodes_jsonl(path, episodes)

    loaded = json.loads(path.read_text().splitlines()[0])
    assert loaded["smoothness"] == pytest.approx(0.4242)


def test_write_episodes_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "episodes.jsonl"
    write_episodes_jsonl(path, [_episode()])
    assert path.exists()


def test_write_summary_csv_row_keys_match_summary_csv_fields_exactly(tmp_path):
    episodes = [_episode(trial=0), _episode(trial=1, success=False)]
    row = write_summary_csv(
        tmp_path / "summary.csv", episodes, _fake_metadata(), seed=7, vram_peak_gb=8.5
    )
    assert set(row.keys()) == set(SUMMARY_CSV_FIELDS)


def test_write_summary_csv_computed_fields(tmp_path):
    episodes = [_episode(trial=0, success=True), _episode(trial=1, success=False)]
    row = write_summary_csv(
        tmp_path / "summary.csv", episodes, _fake_metadata(), seed=7, vram_peak_gb=8.5
    )
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["n_episodes"] == 2
    assert row["suite"] == "libero_spatial"  # taken from episodes[0]
    assert row["model_id"] == "oft"
    assert row["vram_peak_gb"] == 8.5
    assert row["seed"] == 7


def test_write_summary_csv_empty_episodes_zero_success_rate(tmp_path):
    row = write_summary_csv(
        tmp_path / "summary.csv", [], _fake_metadata(), seed=7, vram_peak_gb=0.0
    )
    assert row["success_rate"] == 0.0
    assert row["n_episodes"] == 0
    assert row["suite"] == ""


def test_write_summary_csv_all_fail_nan_smoothness_becomes_empty_cell(tmp_path):
    # NaN summary values are mapped to None (nan_to_none), same convention as
    # write_episodes_jsonl -- csv.DictWriter renders None as an empty cell, not "nan".
    episodes = [_episode(success=False, smoothness=float("nan"))]
    path = tmp_path / "summary.csv"
    write_summary_csv(path, episodes, _fake_metadata(), seed=7, vram_peak_gb=8.5)

    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["mean_succ_smoothness"] == ""


def test_write_summary_csv_header_round_trips_via_dictreader(tmp_path):
    episodes = [_episode()]
    path = tmp_path / "summary.csv"
    row = write_summary_csv(
        path, episodes, _fake_metadata(), seed=7, vram_peak_gb=8.5
    )
    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SUMMARY_CSV_FIELDS
        read_row = next(reader)
    assert read_row["model_id"] == row["model_id"]
    assert float(read_row["success_rate"]) == pytest.approx(row["success_rate"])


def test_write_all_suites_csv_two_rows_preserve_order(tmp_path):
    row_a = {field: 1 for field in SUMMARY_CSV_FIELDS}
    row_a["suite"] = "libero_spatial"
    row_b = {field: 2 for field in SUMMARY_CSV_FIELDS}
    row_b["suite"] = "libero_object"
    path = tmp_path / "all_suites.csv"

    write_all_suites_csv(path, [row_a, row_b])

    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["suite"] == "libero_spatial"
    assert rows[1]["suite"] == "libero_object"


def test_write_all_suites_csv_uses_same_field_order_as_summary_csv(tmp_path):
    row = {field: "x" for field in SUMMARY_CSV_FIELDS}
    path = tmp_path / "all_suites.csv"
    write_all_suites_csv(path, [row])
    with open(path) as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SUMMARY_CSV_FIELDS
