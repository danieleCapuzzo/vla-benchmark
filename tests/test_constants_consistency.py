"""
Guards against the exact duplication bug this file was added to fix: each wrapper used to
define its own NUM_CAMERAS constant *and* run_eval.py kept a parallel per-model dict, with
nothing forcing them to agree. Now both derive from config/constants.py -- these tests assert
that derivation actually happened, not just that the numbers currently match by coincidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import NUM_CAMERAS_BY_MODEL, RENDER_RESOLUTION_BY_MODEL


def test_num_cameras_by_model_covers_every_wrapper():
    assert set(NUM_CAMERAS_BY_MODEL) == {"openvla", "oft", "mail"}


def test_render_resolution_by_model_covers_every_wrapper():
    assert set(RENDER_RESOLUTION_BY_MODEL) == {"openvla", "oft", "mail"}


def test_wrapper_num_cameras_is_the_same_object_as_constants():
    from wrappers.mail_wrapper import NUM_CAMERAS as mail_cams
    from wrappers.openvla_wrapper import NUM_CAMERAS as openvla_cams
    from wrappers.oft_wrapper import NUM_CAMERAS as oft_cams

    assert openvla_cams == NUM_CAMERAS_BY_MODEL["openvla"]
    assert oft_cams == NUM_CAMERAS_BY_MODEL["oft"]
    assert mail_cams == NUM_CAMERAS_BY_MODEL["mail"]


def test_run_eval_num_cameras_by_model_is_the_shared_dict():
    import run_eval

    assert run_eval.NUM_CAMERAS_BY_MODEL is NUM_CAMERAS_BY_MODEL


def test_mail_trained_at_128_openvla_family_at_256():
    # the render-resolution fix this file exists for: MaIL was trained on native-128 LIBERO
    # hdf5 demos and must be evaluated at that resolution, not the OpenVLA-family 256.
    assert RENDER_RESOLUTION_BY_MODEL["mail"] == 128
    assert RENDER_RESOLUTION_BY_MODEL["openvla"] == 256
    assert RENDER_RESOLUTION_BY_MODEL["oft"] == 256
