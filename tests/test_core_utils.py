"""Unit tests for core/core_utils.py: quat2axisangle, gripper action helpers, image extraction."""

import math

import numpy as np
import pytest

from core.core_utils import (
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    invert_gripper_action,
    normalize_gripper_action,
    quat2axisangle,
)


def test_quat2axisangle_identity_quaternion_returns_zeros():
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    result = quat2axisangle(quat)
    assert result == pytest.approx([0.0, 0.0, 0.0])


def test_quat2axisangle_known_90_degree_rotation_about_z():
    # 90-degree rotation about z: quat (x,y,z,w) = (0,0,sin(45deg),cos(45deg))
    half = math.pi / 4
    quat = np.array([0.0, 0.0, math.sin(half), math.cos(half)])
    result = quat2axisangle(quat)
    assert result == pytest.approx([0.0, 0.0, math.pi / 2], abs=1e-6)


def test_quat2axisangle_mutates_input_in_place_on_clip_branch():
    # Documents a real bug-shaped edge: quat2axisangle clips quat[3] in place on the
    # caller's array rather than working on a copy (core_utils.py:68-71).
    quat = np.array([0.0, 0.0, 0.0, 1.5])
    quat2axisangle(quat)
    assert quat[3] == pytest.approx(1.0)


def test_quat2axisangle_negative_out_of_range_clips_to_minus_one():
    quat = np.array([0.0, 0.0, 0.0, -1.5])
    quat2axisangle(quat)
    assert quat[3] == pytest.approx(-1.0)


def test_normalize_gripper_action_remaps_zero_to_one_range():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = normalize_gripper_action(action, binarize=False)
    assert result[-1] == pytest.approx(-1.0)

    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    result = normalize_gripper_action(action, binarize=False)
    assert result[-1] == pytest.approx(1.0)


def test_normalize_gripper_action_binarize_true_uses_sign():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8])
    result = normalize_gripper_action(action, binarize=True)
    assert result[-1] == pytest.approx(1.0)

    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    result = normalize_gripper_action(action, binarize=True)
    assert result[-1] == pytest.approx(-1.0)


def test_normalize_gripper_action_boundary_0_5_maps_to_zero():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    unbinarized = normalize_gripper_action(action, binarize=False)
    assert unbinarized[-1] == pytest.approx(0.0)

    binarized = normalize_gripper_action(action, binarize=True)
    assert binarized[-1] == pytest.approx(0.0)  # np.sign(0.0) == 0.0, not +/-1


def test_normalize_gripper_action_does_not_mutate_input():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    original = action.copy()
    normalize_gripper_action(action)
    assert np.array_equal(action, original)


def test_invert_gripper_action_negates_last_dim():
    action = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    result = invert_gripper_action(action)
    assert result[-1] == pytest.approx(-1.0)
    assert result[:-1] == pytest.approx(action[:-1])


def test_invert_gripper_action_does_not_mutate_input():
    action = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    original = action.copy()
    invert_gripper_action(action)
    assert np.array_equal(action, original)


def test_get_libero_image_selects_agentview_key_and_rotates_180():
    img = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    obs = {"agentview_image": img}
    result = get_libero_image(obs)
    assert np.array_equal(result, img[::-1, ::-1])


def test_get_libero_wrist_image_selects_eye_in_hand_key_and_rotates_180():
    img = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    obs = {"robot0_eye_in_hand_image": img}
    result = get_libero_wrist_image(obs)
    assert np.array_equal(result, img[::-1, ::-1])


def test_get_libero_dummy_action_is_no_op_with_closed_gripper():
    assert get_libero_dummy_action("openvla") == [0, 0, 0, 0, 0, 0, -1]
