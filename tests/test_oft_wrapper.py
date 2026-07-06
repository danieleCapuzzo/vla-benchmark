"""Unit test for OftWrapper.predict() output shape/dtype.

Approach: mock `get_vla_action` rather than running against the real checkpoint. This
keeps the unit suite fast, GPU-free, and independent of network access / HF downloads --
suitable for CI. The real end-to-end path (checkpoint load, actual inference, VRAM) is
exercised separately via `python run_eval.py --model oft --suite spatial --quant 8bit
--rollouts 2`, which is the appropriate place to validate real-model behavior since it
also covers the full LIBERO env integration that a wrapper-only mock cannot.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _install_fake_oft_modules():
    """Installs minimal fake `experiments.*` / `prismatic.*` modules on sys.modules.

    The real openvla-oft package requires a GPU-capable torch build plus LIBERO installed
    to even import (chain import failures otherwise). Since this test only needs to verify
    OftWrapper's adaptation logic (resize -> get_vla_action -> PolicyOutput), we fake out
    just the symbols oft_wrapper.py imports, and patch `get_vla_action` per-test.
    """
    experiments_pkg = types.ModuleType("experiments")
    robot_pkg = types.ModuleType("experiments.robot")
    libero_pkg = types.ModuleType("experiments.robot.libero")
    run_libero_eval_mod = types.ModuleType("experiments.robot.libero.run_libero_eval")
    openvla_utils_mod = types.ModuleType("experiments.robot.openvla_utils")
    prismatic_pkg = types.ModuleType("prismatic")
    vla_pkg = types.ModuleType("prismatic.vla")
    constants_mod = types.ModuleType("prismatic.vla.constants")

    class _FakeGenerateConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    run_libero_eval_mod.GenerateConfig = _FakeGenerateConfig

    openvla_utils_mod.get_vla = MagicMock()
    openvla_utils_mod.get_processor = MagicMock()
    openvla_utils_mod.get_action_head = MagicMock()
    openvla_utils_mod.get_proprio_projector = MagicMock()
    openvla_utils_mod.get_vla_action = MagicMock()
    openvla_utils_mod.resize_image_for_policy = lambda img, size: img

    constants_mod.NUM_ACTIONS_CHUNK = 8
    constants_mod.ACTION_DIM = 7
    constants_mod.PROPRIO_DIM = 8

    modules = {
        "experiments": experiments_pkg,
        "experiments.robot": robot_pkg,
        "experiments.robot.libero": libero_pkg,
        "experiments.robot.libero.run_libero_eval": run_libero_eval_mod,
        "experiments.robot.openvla_utils": openvla_utils_mod,
        "prismatic": prismatic_pkg,
        "prismatic.vla": vla_pkg,
        "prismatic.vla.constants": constants_mod,
    }
    sys.modules.update(modules)
    return openvla_utils_mod


@pytest.fixture
def fake_oft_modules():
    saved = {k: sys.modules.get(k) for k in list(sys.modules) if k.startswith(("experiments", "prismatic"))}
    mod = _install_fake_oft_modules()
    yield mod
    for k in list(sys.modules):
        if k.startswith(("experiments", "prismatic")):
            del sys.modules[k]
    for k, v in saved.items():
        if v is not None:
            sys.modules[k] = v


def test_oft_wrapper_predict_shape_and_dtype(fake_oft_modules):
    from wrappers.oft_wrapper import OftWrapper

    fake_chunk = [np.random.randn(7).astype(np.float32) for _ in range(8)]
    fake_oft_modules.get_vla_action.return_value = fake_chunk

    fake_vla = MagicMock()
    fake_vla.llm_dim = 4096
    fake_vla.parameters.return_value = iter([])
    fake_oft_modules.get_vla.return_value = fake_vla

    wrapper = OftWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial_no_noops")
    with patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.synchronize"):
        wrapper.load(quant="none", num_cameras=1)

        observation = {
            "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
        }
        output = wrapper.predict(observation, "pick up the object")

    assert output.actions.shape == (8, 7)
    assert output.actions.dtype == np.float32
    assert output.is_chunked is True
    assert output.raw_latency_s >= 0.0
