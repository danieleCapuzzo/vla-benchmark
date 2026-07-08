"""Unit tests for run_eval.py's pure CLI helpers: resolve_suites, _resolve_run_dir,
default_config_for.

Importing run_eval triggers the real wrapper registration imports (openvla/oft/mail), which
succeed without a GPU call in .oft-env -- no mocking needed for these pure functions.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_eval  # noqa: E402


def test_resolve_suites_all_expands_to_full_list():
    assert run_eval.resolve_suites(["all"]) == run_eval.ALL_SUITES


def test_resolve_suites_dedup_preserves_first_seen_order():
    result = run_eval.resolve_suites(["object", "spatial", "object", "goal"])
    assert result == ["object", "spatial", "goal"]


def test_resolve_suites_all_mixed_with_others_still_expands_to_all():
    result = run_eval.resolve_suites(["spatial", "all"])
    assert result == run_eval.ALL_SUITES


def test_resolve_suites_single_suite_passthrough():
    assert run_eval.resolve_suites(["spatial"]) == ["spatial"]


def test_resolve_run_dir_fresh_directory(tmp_path):
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1)
    assert result == tmp_path / "oft_8bit_1cam"


def test_resolve_run_dir_collision_numbering_starts_at_two(tmp_path):
    (tmp_path / "oft_8bit_1cam").mkdir()
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1)
    assert result == tmp_path / "oft2_8bit_1cam"


def test_resolve_run_dir_multiple_collisions_increment(tmp_path):
    (tmp_path / "oft_8bit_1cam").mkdir()
    (tmp_path / "oft2_8bit_1cam").mkdir()
    (tmp_path / "oft3_8bit_1cam").mkdir()
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1)
    assert result == tmp_path / "oft4_8bit_1cam"


def test_default_config_for_known_models():
    assert run_eval.default_config_for("openvla").name == "config_openvla.json"
    assert run_eval.default_config_for("oft").name == "config_oft.json"


def test_default_config_for_mail_falls_back_to_oft_config():
    assert run_eval.default_config_for("mail").name == "config_oft.json"


def test_default_config_for_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        run_eval.default_config_for("does_not_exist")
