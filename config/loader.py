"""
Loader for the testbench JSON configuration file.
Centralizes and preprocesses all run parameters according to the format

"suites": {
    "suite_name": {
        "checkpoint": something
        "unnorm_key": something
        "task_suite_name": something
    },
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config_oft.json"

_NO_NOOPS_SUFFIX = "_no_noops"

# frozen value for the evaluation protocol (see doc/ARCHITECTURE.md "20 rollout/task").
# any override is NON-PROTOCOL (smoke tests only) and will be highlighted to the user.
FROZEN_ROLLOUTS_PER_TASK = 20


@dataclass
class SuiteConfig:
    """
    Static configuration for a single LIBERO suite.

    Attributes:
        checkpoint: HuggingFace Hub id (or local path) of the model checkpoint for this suite.
        unnorm_key: Action un-normalization key used by ``get_vla_action``.
        task_suite_name: LIBERO benchmark task suite identifier (e.g. "libero_spatial").
    """

    checkpoint: str
    unnorm_key: str
    task_suite_name: str


@dataclass
class RunConfig:
    """
    Top-level run configuration, loaded from ``config/config.json``.

    Attributes:
        rollouts_per_task: number of rollouts per task.
        num_cameras: number of cameras fed to the policy (1 or 2).
                     CAVEAT for OFT, the checkpoints were fine-tuned with 2 cameras (agentview+wrist)
                     so single camera is supported but completely out of distribution.
        quant: quantization mode ("none", "8bit", "4bit").
        seed: random seed for reproducibility.
        num_steps_wait: number of no-op steps at episode start to let objects settle.
        output_dir: base directory where per-run result folders are written.
        suites: mapping from suite name ("spatial", "object", "goal", "long") to its `SuiteConfig`.
    """

    rollouts_per_task: int = FROZEN_ROLLOUTS_PER_TASK
    num_cameras: int = 1
    quant: str = "none"
    seed: int = 7
    num_steps_wait: int = 10
    output_dir: str = "results"
    suites: Dict[str, SuiteConfig] = field(default_factory=dict)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RunConfig:
    """
    Loads the run configuration from a JSON file.

    Args:
        path: path to the JSON config file, defaults to `config/config_oft.json`.

    Returns:
        a populated `RunConfig` class.

    Raises:
        FileNotFoundError: if the config file does not exist.
        KeyError: if a suite entry is missing a required field.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = json.load(f)

    suites = {
        name: SuiteConfig(
            checkpoint=entry["checkpoint"],
            unnorm_key=entry["unnorm_key"],
            task_suite_name=entry["task_suite_name"],
        )
        for name, entry in raw.get("suites", {}).items()
    }

    return RunConfig(
        rollouts_per_task=raw.get("rollouts_per_task", FROZEN_ROLLOUTS_PER_TASK),
        num_cameras=raw.get("num_cameras", 1),
        quant=raw.get("quant", "none"),
        seed=raw.get("seed", 7),
        num_steps_wait=raw.get("num_steps_wait", 10),
        output_dir=raw.get("output_dir", "results"),
        suites=suites,
    )


def resolve_unnorm_key(unnorm_key: str, available_keys: Iterable[str]) -> str:
    """
    Strips the mistaken _no_noops suffix on unnorm_key

    Args:
        unnorm_key: `unnorm_key` as configured (from `SuiteConfig`).
        available_keys: the checkpoint's actual norm_stats keys.

    Returns:
        `unnorm_key` unchanged if already present in `available_keys`, otherwise the
        `_no_noops`-toggled variant if that one is present instead.
    """
    available_keys = set(available_keys)
    if unnorm_key in available_keys:
        return unnorm_key
    if unnorm_key.endswith(_NO_NOOPS_SUFFIX):
        stripped = unnorm_key[: -len(_NO_NOOPS_SUFFIX)]
        if stripped in available_keys:
            return stripped
    elif f"{unnorm_key}{_NO_NOOPS_SUFFIX}" in available_keys:
        return f"{unnorm_key}{_NO_NOOPS_SUFFIX}"
    return unnorm_key
