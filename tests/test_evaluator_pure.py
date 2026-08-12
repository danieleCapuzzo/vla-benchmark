"""Unit tests for the CPU-only slices of core/evaluator.py::LiberoEvaluator: __init__'s
TASK_MAX_STEPS lookup and the _build_observation staticmethod. The rollout loop itself
(`run`, `_run_episode`) requires a real LIBERO/MuJoCo env and is exercised only via
`run_eval.py`, not unit tests.
"""

import numpy as np
import pytest

from config.constants import DEFAULT_RENDER_RESOLUTION, TASK_MAX_STEPS
from core.evaluator import LiberoEvaluator


def test_init_known_suite_sets_max_steps():
    evaluator = LiberoEvaluator("libero_spatial")
    assert evaluator._max_steps == TASK_MAX_STEPS["libero_spatial"]


def test_init_unknown_suite_raises_keyerror():
    with pytest.raises(KeyError):
        LiberoEvaluator("libero_does_not_exist")


def test_init_default_render_resolution():
    evaluator = LiberoEvaluator("libero_spatial")
    assert evaluator._render_resolution == DEFAULT_RENDER_RESOLUTION


def test_init_render_resolution_override():
    evaluator = LiberoEvaluator("libero_spatial", render_resolution=128)
    assert evaluator._render_resolution == 128


def test_build_observation_key_set_and_state_concatenation_order():
    obs = {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.ones((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    result = LiberoEvaluator._build_observation(obs)

    assert set(result.keys()) == {"full_image", "wrist_image", "state", "raw_obs"}
    # state = eef_pos (3) + quat2axisangle(eef_quat) (3) + gripper_qpos (2) = 8
    assert result["state"].shape == (8,)
    assert result["state"][:3] == pytest.approx([1.0, 2.0, 3.0])
    assert result["state"][3:6] == pytest.approx([0.0, 0.0, 0.0])  # identity quat -> zeros
    assert result["state"][6:] == pytest.approx([0.04, -0.04])


def test_build_observation_raw_obs_carried_through_unchanged():
    obs = {
        "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    result = LiberoEvaluator._build_observation(obs)
    assert result["raw_obs"] is obs
