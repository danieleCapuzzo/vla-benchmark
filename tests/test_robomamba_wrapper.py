"""
CPU-only unit tests for RoboMamba's action head and preprocessing. No GPU or checkpoint
needed -- see tests/test_core_utils_conversions.py for the shared image/action conversions,
and tests/test_wrapper_validation.py for the load()/get_metadata() contract checks.
"""

import inspect

import numpy as np
import pytest
import torch

from config.constants import OPENVLA_INPUT_RESOLUTION
from config.paths import LIBERO_DATASETS, ROBOMAMBA_SRC
from core.arch.robomamba_arch import (
    ACTION_CHUNK,
    ACTION_DIM,
    HEAD_BLOCKS,
    HEAD_WIDTH,
    HIDDEN_DIM,
    IMAGE_SIZE,
    RoboMambaHead,
    _MLPResBlock,
    to_model_input,
)
from core.dataproc.libero_dataset import _chunk_actions, _task_instructions

IN_DIM = HIDDEN_DIM


def test_head_output_shape():
    head = RoboMambaHead()
    features = torch.randn(4, IN_DIM)
    out = head(features)
    assert out.shape == (4, ACTION_CHUNK, ACTION_DIM)


def test_head_layer_shapes():
    head = RoboMambaHead()
    assert isinstance(head.stem[0], torch.nn.LayerNorm)
    assert head.stem[0].normalized_shape == (IN_DIM,)
    assert head.stem[1].in_features == IN_DIM and head.stem[1].out_features == HEAD_WIDTH
    assert len(head.blocks) == HEAD_BLOCKS
    assert head.out[1].in_features == HEAD_WIDTH
    assert head.out[1].out_features == ACTION_CHUNK * ACTION_DIM


def test_head_blocks_are_residual():
    # a zeroed block body must pass its input through untouched
    block = _MLPResBlock(HEAD_WIDTH)
    for p in block.body.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(3, HEAD_WIDTH)
    assert torch.allclose(block(x), x)


def test_chunk_actions_is_unclamped_mid_demo():
    n = ACTION_CHUNK + 3
    demo_actions = np.arange(n, dtype=np.float32).reshape(n, 1).repeat(ACTION_DIM, axis=1)

    chunk = _chunk_actions(demo_actions, 0, ACTION_CHUNK)
    assert chunk.shape == (ACTION_CHUNK, ACTION_DIM)
    assert chunk[:, 0].tolist() == [float(i) for i in range(ACTION_CHUNK)]


def test_chunk_actions_clamps_at_demo_end():
    # the last chunk repeats the demo's final action instead of running past it
    n = ACTION_CHUNK + 3
    demo_actions = np.arange(n, dtype=np.float32).reshape(n, 1).repeat(ACTION_DIM, axis=1)

    chunk = _chunk_actions(demo_actions, n - 1, ACTION_CHUNK)
    assert chunk[:, 0].tolist() == [float(n - 1)] * ACTION_CHUNK


def test_robomamba_src_exists():
    assert ROBOMAMBA_SRC.is_dir(), f"missing vendored robo_mamba source at {ROBOMAMBA_SRC}"


def test_image_size_matches_clip_native_resolution():
    assert IMAGE_SIZE == OPENVLA_INPUT_RESOLUTION


_SPATIAL_DIRS = [LIBERO_DATASETS / "libero_spatial", LIBERO_DATASETS / "libero_spatial_regen"]


@pytest.mark.parametrize("data_dir", _SPATIAL_DIRS, ids=lambda p: p.name)
def test_every_spatial_hdf5_resolves_to_an_instruction(data_dir):
    """Regenerated hdf5s have no problem_info attr -- instructions come from the registry."""
    pytest.importorskip("libero")
    if not any(data_dir.glob("*.hdf5")):
        pytest.skip(f"{data_dir} not present")
    instructions = _task_instructions("spatial")
    for path in sorted(data_dir.glob("*.hdf5")):
        assert path.stem.removesuffix("_demo") in instructions


def test_registry_instruction_matches_shipped_hdf5_problem_info():
    """The invariant the filename-based lookup relies on: registry language must match the
    original hdf5's stored instruction, so the same lookup is valid for regen dirs too."""
    pytest.importorskip("libero")
    import json

    import h5py

    data_dir = LIBERO_DATASETS / "libero_spatial"
    if not any(data_dir.glob("*.hdf5")):
        pytest.skip(f"{data_dir} not present")
    instructions = _task_instructions("spatial")
    for path in sorted(data_dir.glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            stored = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
        assert instructions[path.stem.removesuffix("_demo")] == stored
