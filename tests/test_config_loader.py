"""Unit tests for config/loader.py: resolve_unnorm_key branches, load_config parsing."""

import json

import pytest

from config.loader import RunConfig, SuiteConfig, load_config, resolve_unnorm_key


def test_resolve_unnorm_key_already_present_unchanged():
    assert resolve_unnorm_key("libero_spatial", {"libero_spatial", "libero_object"}) == (
        "libero_spatial"
    )


def test_resolve_unnorm_key_strips_no_noops_suffix_when_stripped_present():
    assert resolve_unnorm_key("libero_spatial_no_noops", {"libero_spatial"}) == (
        "libero_spatial"
    )


def test_resolve_unnorm_key_adds_no_noops_suffix_when_only_suffixed_present():
    assert resolve_unnorm_key("libero_spatial", {"libero_spatial_no_noops"}) == (
        "libero_spatial_no_noops"
    )


def test_resolve_unnorm_key_fallthrough_when_neither_variant_present():
    assert resolve_unnorm_key("libero_spatial", {"libero_object"}) == "libero_spatial"


def test_resolve_unnorm_key_suffixed_key_with_neither_variant_present():
    # ends with _no_noops, stripped key absent -> falls through unchanged (no elif branch
    # taken since it's only reached for keys NOT ending in _no_noops).
    result = resolve_unnorm_key("libero_spatial_no_noops", {"libero_object"})
    assert result == "libero_spatial_no_noops"


def test_load_config_minimal_json_applies_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({}))

    cfg = load_config(path)

    assert isinstance(cfg, RunConfig)
    assert cfg.rollouts_per_task == 20
    assert cfg.quant == "none"
    assert cfg.seed == 7
    assert cfg.num_steps_wait == 10
    assert cfg.output_dir == "results"
    assert cfg.suites == {}


def test_load_config_full_json_parses_suites(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "rollouts_per_task": 5,
                "quant": "8bit",
                "seed": 42,
                "num_steps_wait": 3,
                "output_dir": "custom_results",
                "suites": {
                    "spatial": {
                        "checkpoint": "org/ckpt",
                        "unnorm_key": "libero_spatial",
                        "task_suite_name": "libero_spatial",
                    }
                },
            }
        )
    )

    cfg = load_config(path)

    assert cfg.rollouts_per_task == 5
    assert cfg.quant == "8bit"
    assert cfg.seed == 42
    assert cfg.num_steps_wait == 3
    assert cfg.output_dir == "custom_results"
    assert "spatial" in cfg.suites
    assert cfg.suites["spatial"] == SuiteConfig(
        checkpoint="org/ckpt",
        unnorm_key="libero_spatial",
        task_suite_name="libero_spatial",
    )


def test_load_config_suite_missing_required_field_raises_keyerror(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"suites": {"spatial": {"checkpoint": "org/ckpt"}}})
    )
    with pytest.raises(KeyError):
        load_config(path)


def test_load_config_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.json")
