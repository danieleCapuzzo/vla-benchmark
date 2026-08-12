"""
CPU-only unit tests for RoboMamba's action head and preprocessing. No GPU or checkpoint
needed -- see tests/test_core_utils_conversions.py for the shared image/action conversions,
and tests/test_wrapper_validation.py for the load()/get_metadata() contract checks.
"""

import inspect

import numpy as np
import torch

from core.robomamba_arch import (
    ACTION_CHUNK,
    ACTION_DIM,
    FEATURE_CACHE_VERSION,
    HEAD_BLOCKS,
    HEAD_WIDTH,
    HIDDEN_DIM,
    HISTORY_FRAMES,
    IMAGE_SIZE,
    RoboMambaHead,
    _MLPResBlock,
    build_history_chunks,
    extract_features,
    feature_cache_path,
    to_model_input,
)

IN_DIM = HIDDEN_DIM * HISTORY_FRAMES


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


def test_build_history_chunks_clamps_at_demo_boundaries():
    # two demos, so the builder cannot bleed one demo's frames into the other
    n = ACTION_CHUNK + 3
    ramp = torch.arange(2 * n, dtype=torch.float32).unsqueeze(1)
    features = ramp.repeat(1, HIDDEN_DIM)
    actions = ramp.repeat(1, ACTION_DIM)
    demo_ids = torch.tensor([0] * n + [1] * n)

    x, y, ids = build_history_chunks(features, actions, demo_ids)
    assert x.shape == (2 * n, IN_DIM)
    assert y.shape == (2 * n, ACTION_CHUNK, ACTION_DIM)
    assert ids.tolist() == demo_ids.tolist()

    # t < ACTION_CHUNK clamps history to the demo's own first frame
    assert x[0, 0] == 0.0 and x[0, HIDDEN_DIM] == 0.0
    assert x[ACTION_CHUNK, 0] == 0.0 and x[ACTION_CHUNK, HIDDEN_DIM] == ACTION_CHUNK
    # demo 1 starts over rather than reaching back into demo 0
    assert x[n, 0] == float(n) and x[n, HIDDEN_DIM] == float(n)

    # the last chunk repeats the demo's final action instead of running into demo 1
    assert y[0, :, 0].tolist() == [float(i) for i in range(ACTION_CHUNK)]
    assert y[n - 1, :, 0].tolist() == [float(n - 1)] * ACTION_CHUNK
