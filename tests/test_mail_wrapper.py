"""
CPU-only unit tests for the MaIL wrapper's preprocessing and action-chunk bookkeeping.

`predict()` is exercised against a stub policy/scaler so no checkpoint or GPU is needed --
see tests/test_wrapper_validation.py for the load()/get_metadata() contract checks and
tests/test_core_utils_conversions.py for the shared conversions themselves.
"""

import numpy as np
import pytest
import torch

from config.constants import MAIL_IMAGE_SIZE as IMAGE_SIZE
from config.constants import NUM_CAMERAS_BY_MODEL
from wrappers.mail_wrapper import MailWrapper, _to_model_image

ACTION_SEQ = 5
OBS_SEQ = 5


def _frame(h=256, w=256, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _obs(seed=0):
    return {
        "raw_obs": {
            "agentview_image": _frame(seed=seed),
            "robot0_eye_in_hand_image": _frame(seed=seed + 100),
        }
    }


# ------------------------------------------------------------------ preprocessing


def test_to_model_image_shape_dtype_and_range():
    tensor = _to_model_image(_frame())
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_to_model_image_does_not_rotate():
    """MaIL trains on native-orientation frames; rotating here would put its own encoder
    off-distribution. Guards against someone 'fixing' this to match RoboMamba."""
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[0, 0] = 255  # marker in the top-left
    tensor = _to_model_image(image)
    assert float(tensor[0, 0, 0]) == pytest.approx(1.0), "marker moved: frame was rotated"


def test_to_model_image_preserves_channel_order():
    """HWC -> CHW must permute axes, not reorder colours."""
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 255, 0, 0
    tensor = _to_model_image(image)
    assert float(tensor[0].mean()) == pytest.approx(1.0)
    assert float(tensor[1].mean()) == pytest.approx(0.0)


def test_to_model_image_scales_by_255():
    image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
    np.testing.assert_allclose(_to_model_image(image).numpy(), 1.0)


def test_to_model_image_accepts_native_and_env_resolutions():
    """The env renders at 256; the demos are 128. Both must reach IMAGE_SIZE."""
    for size in (128, 256):
        assert _to_model_image(_frame(size, size)).shape == (3, IMAGE_SIZE, IMAGE_SIZE)


# ------------------------------------------------------------------ predict()


class _StubArch:
    obs_seq = OBS_SEQ
    action_seq = ACTION_SEQ


class _StubPolicy:
    """Stands in for MailPolicy: records what it was called with, returns a fixed chunk."""

    def __init__(self, chunk):
        self.arch = _StubArch()
        self.device = "cpu"
        self.chunk = chunk
        self.calls = []

    def predict(self, agentview, wrist, language=None):
        self.calls.append((agentview.shape, None if wrist is None else wrist.shape, language))
        return torch.from_numpy(self.chunk)

    def parameters(self):
        return iter([torch.zeros(1)])


class _StubScaler:
    """MaIL's ActionScaler at eval only needs to un-scale; identity keeps the test readable."""

    def inverse_scale_output(self, actions):
        return actions


def _wrapper(chunk, language=False):
    w = MailWrapper(checkpoint="unused.pt", unnorm_key="", language=language)
    w._policy = _StubPolicy(chunk)
    w._scaler = _StubScaler()
    w._meta = {"suite": "spatial", "num_cameras": NUM_CAMERAS_BY_MODEL["mail"]}
    return w


def _open_gripper_chunk(n=ACTION_SEQ):
    """A chunk of raw LIBERO-convention actions with the gripper open (-1)."""
    chunk = np.zeros((n, 7), dtype=np.float32)
    chunk[:, -1] = -1.0
    return chunk


def test_predict_returns_single_action_with_correct_shape():
    out = _wrapper(_open_gripper_chunk()).predict(_obs(), "task")
    assert out.actions.shape == (1, 7)
    assert out.is_chunked is False
    assert out.actions.dtype == np.float32


def test_predict_applies_gripper_convention():
    """Raw -1 (open) must leave the wrapper as RLDS 1.0; the evaluator inverts it back."""
    out = _wrapper(_open_gripper_chunk()).predict(_obs(), "task")
    assert out.actions[0, -1] == pytest.approx(1.0)


def test_predict_queries_model_once_per_chunk():
    """Action chunking: the network runs on call 1 and again only after ACTION_SEQ calls."""
    w = _wrapper(_open_gripper_chunk())
    for _ in range(ACTION_SEQ):
        w.predict(_obs(), "task")
    assert len(w._policy.calls) == 1
    w.predict(_obs(), "task")
    assert len(w._policy.calls) == 2


def test_predict_reports_latency_only_on_real_model_calls():
    """Cached steps must report 0 latency, otherwise effective_aps is wrong."""
    w = _wrapper(_open_gripper_chunk())
    first = w.predict(_obs(), "task")
    cached = [w.predict(_obs(), "task") for _ in range(ACTION_SEQ - 1)]
    assert first.raw_latency_s >= 0.0
    assert all(o.raw_latency_s == 0.0 for o in cached)


def test_predict_walks_through_the_chunk_in_order():
    chunk = np.tile(np.arange(ACTION_SEQ, dtype=np.float32).reshape(-1, 1), (1, 7))
    chunk[:, -1] = -1.0
    w = _wrapper(chunk)
    xs = [float(w.predict(_obs(), "task").actions[0, 0]) for _ in range(ACTION_SEQ)]
    assert xs == list(range(ACTION_SEQ))


def test_predict_does_not_mutate_the_action_cache():
    """env_gripper_to_rlds must copy: the cache is indexed again on later steps, and an
    in-place remap would re-convert the same entry (and corrupt replayed chunks)."""
    w = _wrapper(_open_gripper_chunk())
    w.predict(_obs(), "task")
    np.testing.assert_allclose(w._action_cache[:, -1], -1.0)


def test_observation_window_is_capped_at_obs_seq():
    w = _wrapper(_open_gripper_chunk())
    for i in range(OBS_SEQ + 4):
        w.predict(_obs(seed=i), "task")
    assert len(w._agentview_ctx) == OBS_SEQ
    assert len(w._wrist_ctx) == OBS_SEQ


def test_predict_feeds_both_cameras_with_expected_shapes():
    w = _wrapper(_open_gripper_chunk())
    w.predict(_obs(), "task")
    agentview_shape, wrist_shape, _ = w._policy.calls[0]
    assert agentview_shape == (1, 1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert wrist_shape == (1, 1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_reset_clears_window_and_chunk_cache():
    w = _wrapper(_open_gripper_chunk())
    w.predict(_obs(), "task")
    w.reset()
    assert len(w._agentview_ctx) == 0 and len(w._wrist_ctx) == 0
    assert w._action_cache is None and w._action_counter == 0
    w.predict(_obs(), "task")
    assert len(w._policy.calls) == 2, "reset must force a fresh model call"


def test_language_is_forwarded_only_when_enabled():
    w = _wrapper(_open_gripper_chunk(), language=False)
    w.predict(_obs(), "pick up the bowl")
    assert w._policy.calls[0][2] is None

    w = _wrapper(_open_gripper_chunk(), language=True)
    w.predict(_obs(), "pick up the bowl")
    assert w._policy.calls[0][2] == "pick up the bowl"
