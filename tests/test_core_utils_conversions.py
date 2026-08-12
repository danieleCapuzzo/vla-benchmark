"""
CPU-only tests for the shared observation/action conversions in core/core_utils.py.

These encode the invariants that silently break a whole evaluation run when violated: the
gripper-convention round trip (an inverted gripper never picks anything up, but every episode
still completes), image orientation (must match between training and eval), and action
(de)normalization.
"""

import numpy as np
import pytest

from core.core_utils import (
    denormalize_actions,
    env_gripper_to_rlds,
    get_libero_image,
    get_libero_wrist_image,
    invert_gripper_action,
    normalize_actions,
    normalize_gripper_action,
    resize_frame,
    to_upright,
)

ACTION_DIM = 7


def _frame(h=128, w=128, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------ orientation


def test_to_upright_matches_the_original_slice_expression():
    """The OpenVLA/OFT path was `img[::-1, ::-1]` inline; refactoring to `to_upright` must
    not have changed the pixels, or two working models silently go off-distribution."""
    image = _frame()
    np.testing.assert_array_equal(to_upright(image), image[::-1, ::-1])


def test_to_upright_is_its_own_inverse():
    image = _frame()
    np.testing.assert_array_equal(to_upright(to_upright(image)), image)


def test_to_upright_returns_contiguous_array():
    """`[::-1, ::-1]` alone is a negative-stride view, which PIL and torch both reject."""
    assert to_upright(_frame()).flags["C_CONTIGUOUS"]


def test_to_upright_does_not_mutate_input():
    image = _frame()
    original = image.copy()
    to_upright(image)
    np.testing.assert_array_equal(image, original)


def test_to_upright_rotates_rather_than_mirrors():
    """A 180 rotation, not a single flip: a corner marker must travel diagonally."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, 0] = 255
    rotated = to_upright(image)
    assert rotated[-1, -1, 0] == 255
    assert rotated[0, 0, 0] == 0 and rotated[0, -1, 0] == 0 and rotated[-1, 0, 0] == 0


def test_to_upright_preserves_channel_order():
    """Must not flip RGB->BGR; only the spatial axes are reversed."""
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    rotated = to_upright(image)
    np.testing.assert_array_equal(rotated[..., 0], 10)
    np.testing.assert_array_equal(rotated[..., 1], 20)
    np.testing.assert_array_equal(rotated[..., 2], 30)


@pytest.mark.parametrize(
    "getter,key",
    [(get_libero_image, "agentview_image"), (get_libero_wrist_image, "robot0_eye_in_hand_image")],
)
def test_libero_image_getters_rotate_the_right_key(getter, key):
    obs = {"agentview_image": _frame(seed=1), "robot0_eye_in_hand_image": _frame(seed=2)}
    np.testing.assert_array_equal(getter(obs), obs[key][::-1, ::-1])


# ------------------------------------------------------------------ resizing


def test_resize_frame_is_a_noop_at_target_size():
    image = _frame(128, 128)
    assert resize_frame(image, 128) is image, "should avoid a pointless copy"


def test_resize_frame_downsamples_to_square():
    assert resize_frame(_frame(256, 256), 128).shape == (128, 128, 3)


def test_resize_frame_preserves_dtype():
    assert resize_frame(_frame(256, 256), 128).dtype == np.uint8


def test_resize_frame_area_averaging_preserves_flat_regions():
    """INTER_AREA averages the source footprint, so a constant image survives exactly."""
    image = np.full((256, 256, 3), 77, dtype=np.uint8)
    np.testing.assert_array_equal(resize_frame(image, 128), np.full((128, 128, 3), 77))


# ------------------------------------------------------------------ gripper convention


@pytest.mark.parametrize("raw,expected", [(-1.0, 1.0), (1.0, 0.0), (0.0, 0.5)])
def test_env_gripper_to_rlds_maps_the_convention(raw, expected):
    action = np.array([0.0] * 6 + [raw], dtype=np.float32)
    assert env_gripper_to_rlds(action)[-1] == pytest.approx(expected)


def test_env_gripper_to_rlds_does_not_mutate_input():
    """MaIL indexes into a reused action cache; in-place mutation would corrupt the chunk."""
    action = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
    env_gripper_to_rlds(action)
    assert action[-1] == -1.0


def test_env_gripper_to_rlds_leaves_other_dims_untouched():
    action = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0], dtype=np.float32)
    np.testing.assert_allclose(env_gripper_to_rlds(action)[:6], action[:6])


@pytest.mark.parametrize("raw_gripper", [-1.0, 1.0])
def test_full_gripper_chain_is_sign_preserving(raw_gripper):
    """The wrapper's remap exists ONLY to pre-compensate the evaluator's OpenVLA-oriented
    post-processing. Composed, the chain must be the identity in sign -- otherwise the
    gripper is inverted for the entire episode and nothing can ever be picked up. Verified
    end-to-end against ground-truth LIBERO demo replays."""
    action = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, raw_gripper], dtype=np.float32)
    sent = invert_gripper_action(
        normalize_gripper_action(env_gripper_to_rlds(action), binarize=True)
    )
    assert np.sign(sent[-1]) == np.sign(raw_gripper)
    np.testing.assert_allclose(sent[:6], action[:6])


def test_skipping_the_remap_inverts_the_gripper():
    """Guards the regression directly: feeding raw LIBERO actions to the evaluator WITHOUT
    env_gripper_to_rlds flips the gripper. This is what made GT demo replays fail."""
    for raw_gripper in (-1.0, 1.0):
        action = np.array([0.0] * 6 + [raw_gripper], dtype=np.float32)
        sent = invert_gripper_action(normalize_gripper_action(action, binarize=True))
        assert np.sign(sent[-1]) == -np.sign(raw_gripper)


def test_gripper_chain_handles_batched_actions():
    """All the gripper helpers index with `..., -1`, so a chunk (K, 7) must work too."""
    chunk = np.tile(np.array([0.0] * 6 + [-1.0], dtype=np.float32), (8, 1))
    sent = invert_gripper_action(
        normalize_gripper_action(env_gripper_to_rlds(chunk), binarize=True)
    )
    assert sent.shape == (8, ACTION_DIM)
    np.testing.assert_allclose(sent[:, -1], -1.0)


def test_normalize_gripper_action_binarize_false_keeps_continuous_values():
    action = np.array([0.0] * 6 + [0.75], dtype=np.float32)
    assert normalize_gripper_action(action, binarize=False)[-1] == pytest.approx(0.5)


# ------------------------------------------------------------------ action normalization


def test_action_normalization_round_trips():
    actions = np.random.default_rng(0).normal(size=(64, ACTION_DIM)).astype(np.float32)
    mean, std = actions.mean(0), actions.std(0)
    restored = denormalize_actions(normalize_actions(actions, mean, std), mean, std)
    np.testing.assert_allclose(restored, actions, rtol=1e-5, atol=1e-6)


def test_action_normalization_equalizes_dim_scales():
    """The point of normalizing: LIBERO's rotation dims are ~10x smaller than xyz, so raw
    targets let xyz/gripper dominate an unweighted L1 loss and the rotation dims go
    effectively unlearned."""
    scales = np.array([0.42, 0.35, 0.51, 0.038, 0.072, 0.058, 1.0], dtype=np.float32)
    actions = np.random.default_rng(0).normal(size=(4096, ACTION_DIM)).astype(np.float32) * scales
    mean, std = actions.mean(0), actions.std(0)

    assert actions.std(0).max() / actions.std(0).min() > 10  # the problem
    np.testing.assert_allclose(
        normalize_actions(actions, mean, std).std(0), np.ones(ACTION_DIM), rtol=1e-4
    )  # the fix


def test_identity_stats_are_a_noop():
    """--raw-actions passes mean=0/std=1; that path must not alter the targets."""
    actions = np.random.default_rng(0).normal(size=(16, ACTION_DIM)).astype(np.float32)
    zero, one = np.zeros(ACTION_DIM, np.float32), np.ones(ACTION_DIM, np.float32)
    np.testing.assert_allclose(normalize_actions(actions, zero, one), actions)
    np.testing.assert_allclose(denormalize_actions(actions, zero, one), actions)


def test_denormalize_then_gripper_remap_is_the_documented_order():
    """core_utils documents denormalize -> env_gripper_to_rlds. The remap assumes a raw-unit
    gripper, so doing it before denormalization gives a different (wrong) command."""
    mean = np.array([0.0] * 6 + [0.1], dtype=np.float32)
    std = np.array([1.0] * 6 + [0.9], dtype=np.float32)
    predicted = np.array([0.0] * 6 + [-1.0], dtype=np.float32)  # standardized units

    correct = env_gripper_to_rlds(denormalize_actions(predicted, mean, std))
    wrong = denormalize_actions(env_gripper_to_rlds(predicted), mean, std)
    assert correct[-1] != pytest.approx(wrong[-1])
    # raw gripper is 0.1 + 0.9*(-1) = -0.8 (open) -> RLDS 0.9, and stays "open" after the chain
    assert correct[-1] == pytest.approx(0.9)
    assert np.sign(invert_gripper_action(normalize_gripper_action(correct))[-1]) == -1.0
