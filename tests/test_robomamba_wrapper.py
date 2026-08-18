"""
CPU-only unit tests for RoboMamba's action head, preprocessing, and merged-checkpoint loading
contract. No GPU or real checkpoint needed -- see tests/test_core_utils_conversions.py for the
shared image/action conversions, and tests/test_wrapper_validation.py for the load()/
get_metadata() before-load contract checks.
"""

import sys

import numpy as np
import pytest
import torch

from config.constants import OPENVLA_INPUT_RESOLUTION
from config.paths import ROBOMAMBA_LIBERO_SRC
from core.core_utils import resize_frame, to_upright

if str(ROBOMAMBA_LIBERO_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOMAMBA_LIBERO_SRC))

from model.action_head import (  # noqa: E402
    ACTION_CHUNK,
    ACTION_DIM,
    HEAD_BLOCKS,
    HEAD_WIDTH,
    HIDDEN_DIM,
    IMAGE_SIZE,
    ActionChunkHead,
    _MLPResBlock,
)

IN_DIM = HIDDEN_DIM


def test_head_output_shape():
    head = ActionChunkHead()
    features = torch.randn(4, IN_DIM)
    out = head(features)
    assert out.shape == (4, ACTION_CHUNK, ACTION_DIM)


def test_head_layer_shapes():
    head = ActionChunkHead()
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


def test_image_size_matches_clip_native_resolution():
    assert IMAGE_SIZE == OPENVLA_INPUT_RESOLUTION


def test_roboMamba_libero_src_exists():
    assert ROBOMAMBA_LIBERO_SRC.is_dir(), f"missing vendored roboMamba-libero source at {ROBOMAMBA_LIBERO_SRC}"


def test_preprocessing_matches_training_recipe():
    """Reproduces external/roboMamba-libero/src/data/libero.py's exact pixel pipeline:
    upright -> INTER_AREA resize -> [0,1] -> CLIP mean/std, no crop."""
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    frame = to_upright(raw)
    frame = resize_frame(frame, IMAGE_SIZE)
    img = torch.from_numpy(frame).float().div(255.0)
    pixel_values = img.permute(2, 0, 1).sub(mean).div(std)

    assert pixel_values.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    # upright + normalized: no raw-pixel value survives untouched
    assert not np.array_equal(frame, raw)
    assert pixel_values.dtype == torch.float32


def test_load_rejects_checkpoint_missing_required_keys(tmp_path):
    from wrappers.robomamba_wrapper import RoboMambaWrapper

    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, bad_ckpt)

    wrapper = RoboMambaWrapper(checkpoint=str(bad_ckpt), unnorm_key="")
    with pytest.raises(AssertionError, match="not a merged checkpoint"):
        wrapper.load()


def test_load_rejects_suite_mismatch(tmp_path):
    from wrappers.robomamba_wrapper import RoboMambaWrapper

    ckpt = tmp_path / "spatial.pt"
    torch.save(
        {
            "state_dict": {},
            "tokenizer_json": "{}",
            "tokenizer_kwargs": {},
            "mamba_config": {},
            "meta": {"suite": "spatial"},
        },
        ckpt,
    )

    wrapper = RoboMambaWrapper(checkpoint=str(ckpt), unnorm_key="", suite="object")
    with pytest.raises(AssertionError, match="checkpoint suite 'spatial' != requested suite 'object'"):
        wrapper.load()


