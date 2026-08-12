"""
Unit tests for run_eval.py's pure CLI helpers: resolve_suites, _resolve_run_dir,
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
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1, ["spatial"])
    assert result == tmp_path / "oft_8bit_1cam"


def test_resolve_run_dir_reuses_existing_dir_for_a_new_suite(tmp_path):
    """An existing run_dir with results for other suites is reused (not forked into a *2
    sibling) as long as none of the suites about to be run already have results in it --
    e.g. oft_8bit_1cam/long,object exist; evaluating spatial should land in oft_8bit_1cam."""
    (tmp_path / "oft_8bit_1cam" / "long").mkdir(parents=True)
    (tmp_path / "oft_8bit_1cam" / "object").mkdir(parents=True)
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1, ["spatial"])
    assert result == tmp_path / "oft_8bit_1cam"


def test_resolve_run_dir_collision_numbering_starts_at_two(tmp_path):
    (tmp_path / "oft_8bit_1cam" / "spatial").mkdir(parents=True)
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1, ["spatial"])
    assert result == tmp_path / "oft2_8bit_1cam"


def test_resolve_run_dir_multiple_collisions_increment(tmp_path):
    (tmp_path / "oft_8bit_1cam" / "spatial").mkdir(parents=True)
    (tmp_path / "oft2_8bit_1cam" / "spatial").mkdir(parents=True)
    (tmp_path / "oft3_8bit_1cam" / "spatial").mkdir(parents=True)
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1, ["spatial"])
    assert result == tmp_path / "oft4_8bit_1cam"


def test_resolve_run_dir_any_colliding_suite_forces_bump(tmp_path):
    """If multiple suites are requested in one invocation and any of them already has
    results in the candidate dir, the whole invocation bumps to a fresh dir (keeps a
    multi-suite run's results together in one place rather than splitting across dirs)."""
    (tmp_path / "oft_8bit_1cam" / "object").mkdir(parents=True)
    result = run_eval._resolve_run_dir(tmp_path, "oft", "8bit", 1, ["spatial", "object"])
    assert result == tmp_path / "oft2_8bit_1cam"


def test_suite_dir_name_language_suffix():
    assert run_eval._suite_dir_name("spatial", False) == "spatial"
    assert run_eval._suite_dir_name("spatial", True) == "spatial_language"


def test_default_config_for_known_models():
    assert run_eval.default_config_for("openvla").name == "config_openvla.json"
    assert run_eval.default_config_for("oft").name == "config_oft.json"
    # assert run_eval.default_config_for("mail").name == "config_mail.json"       

def test_default_config_for_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        run_eval.default_config_for("does_not_exist")
