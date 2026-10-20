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
# bounds_q99: same scheme as OpenVLA-OFT's BOUNDS_Q99 -- per-dim [q01, q99] mapped to [-1, 1].


def test_action_normalization_round_trips():
    actions = np.random.default_rng(0).uniform(-1, 1, size=(64, ACTION_DIM)).astype(np.float32)
    q01, q99 = np.quantile(actions, 0.01, axis=0), np.quantile(actions, 0.99, axis=0)
    restored = denormalize_actions(normalize_actions(actions, q01, q99), q01, q99)
    # points inside [q01, q99] round-trip exactly; the 1% tails saturate at +/-1 and don't.
    inside = np.all((actions >= q01) & (actions <= q99), axis=-1)
    np.testing.assert_allclose(restored[inside], actions[inside], rtol=1e-4, atol=1e-5)


def test_action_normalization_maps_bounds_to_unit_range():
    """The point of bounds normalization: LIBERO's rotation dims are ~10x smaller than xyz, so
    raw targets let xyz/gripper dominate an unweighted L1 loss and the rotation dims go
    effectively unlearned. Mapping each dim's [q01, q99] to [-1, 1] equalizes their scales."""
    scales = np.array([0.42, 0.35, 0.51, 0.038, 0.072, 0.058, 1.0], dtype=np.float32)
    actions = np.random.default_rng(0).normal(size=(4096, ACTION_DIM)).astype(np.float32) * scales
    q01, q99 = np.quantile(actions, 0.01, axis=0), np.quantile(actions, 0.99, axis=0)

    assert (q99 - q01).max() / (q99 - q01).min() > 10  # the problem
    normalized = normalize_actions(actions, q01, q99)
    assert normalized.min() >= -1.0 - 1e-4 and normalized.max() <= 1.0 + 1e-4  # the fix
    np.testing.assert_allclose(
        np.quantile(normalized, 0.01, axis=0), -np.ones(ACTION_DIM), atol=1e-2
    )
    np.testing.assert_allclose(
        np.quantile(normalized, 0.99, axis=0), np.ones(ACTION_DIM), atol=1e-2
    )


def test_identity_bounds_are_a_noop():
    """q01=-1/q99=1 (e.g. for checkpoints without stored action stats) must not alter targets
    already in [-1, 1]."""
    actions = np.random.default_rng(0).uniform(-1, 1, size=(16, ACTION_DIM)).astype(np.float32)
    q01, q99 = -np.ones(ACTION_DIM, np.float32), np.ones(ACTION_DIM, np.float32)
    np.testing.assert_allclose(normalize_actions(actions, q01, q99), actions, atol=1e-6)
    np.testing.assert_allclose(denormalize_actions(actions, q01, q99), actions, atol=1e-6)


def test_out_of_range_actions_saturate_at_the_boundary():
    """Actions outside [q01, q99] (the 1% tails) clip to +/-1 rather than exceeding it --
    matches OpenVLA-OFT's tf.clip_by_value behavior."""
    q01, q99 = np.zeros(ACTION_DIM, np.float32), np.ones(ACTION_DIM, np.float32)
    beyond_high = np.full(ACTION_DIM, 5.0, dtype=np.float32)
    beyond_low = np.full(ACTION_DIM, -5.0, dtype=np.float32)
    np.testing.assert_allclose(normalize_actions(beyond_high, q01, q99), 1.0)
    np.testing.assert_allclose(normalize_actions(beyond_low, q01, q99), -1.0)


def test_normalize_actions_accepts_cuda_tensors():
    """Some callers pass torch.quantile() outputs, which can be CUDA tensors -- np.clip() would
    force a numpy round trip and crash on device tensors, so normalize_actions must use the
    array's own .clip() method instead."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    actions = torch.full((4, ACTION_DIM), 5.0, device="cuda")
    q01 = torch.zeros(ACTION_DIM, device="cuda")
    q99 = torch.ones(ACTION_DIM, device="cuda")
    result = normalize_actions(actions, q01, q99)
    assert result.is_cuda
    torch.testing.assert_close(result, torch.ones_like(result))


def test_denormalize_then_gripper_remap_is_the_documented_order():
    """core_utils documents denormalize -> env_gripper_to_rlds. The remap assumes a raw-unit
    gripper, so doing it before denormalization gives a different (wrong) command."""
    q01 = np.array([0.0] * 6 + [-1.0], dtype=np.float32)
    q99 = np.array([1.0] * 6 + [0.8], dtype=np.float32)
    predicted = np.array([0.0] * 6 + [-1.0], dtype=np.float32)  # bounds-normalized units

    correct = env_gripper_to_rlds(denormalize_actions(predicted, q01, q99))
    wrong = denormalize_actions(env_gripper_to_rlds(predicted), q01, q99)
    assert correct[-1] != pytest.approx(wrong[-1])
    # raw gripper is q01 (predicted=-1 maps to the low bound) = -1.0 (open) -> RLDS 1.0
    assert correct[-1] == pytest.approx(1.0)
    assert np.sign(invert_gripper_action(normalize_gripper_action(correct))[-1]) == -1.0
