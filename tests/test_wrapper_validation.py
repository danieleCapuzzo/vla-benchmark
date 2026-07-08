"""Unit tests for reachable validation branches in wrappers/: bad quant, bad num_cameras,
get_metadata()-before-load(), and the MailWrapper NotImplementedError skeleton.

OpenVlaWrapper's validation raises before any torch.cuda call, so it needs no mocking --
the real module imports cleanly in .oft-env. OftWrapper's module-level imports require the
same sys.modules faking as tests/test_oft_wrapper.py, so this file reuses that fixture.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_oft_wrapper import fake_oft_modules  # noqa: E402,F401 (reused fixture)


# --- OpenVlaWrapper: no mocking needed, validation raises before any GPU call ---


def test_openvla_wrapper_load_bad_quant_raises_value_error():
    from wrappers.openvla_wrapper import OpenVlaWrapper

    wrapper = OpenVlaWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial")
    with pytest.raises(ValueError, match="Unsupported quant"):
        wrapper.load(quant="bogus")


def test_openvla_wrapper_load_num_cameras_not_one_raises_value_error():
    from wrappers.openvla_wrapper import OpenVlaWrapper

    wrapper = OpenVlaWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial")
    with pytest.raises(ValueError, match="single .agentview. camera"):
        wrapper.load(quant="none", num_cameras=2)


def test_openvla_wrapper_get_metadata_before_load_raises_runtime_error():
    from wrappers.openvla_wrapper import OpenVlaWrapper

    wrapper = OpenVlaWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial")
    with pytest.raises(RuntimeError, match="before load"):
        wrapper.get_metadata()


# --- OftWrapper: needs the fake experiments.*/prismatic.* modules to import ---


def test_oft_wrapper_load_bad_quant_raises_value_error(fake_oft_modules):
    from wrappers.oft_wrapper import OftWrapper

    wrapper = OftWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial_no_noops")
    with pytest.raises(ValueError, match="Unsupported quant"):
        wrapper.load(quant="bogus")


def test_oft_wrapper_get_metadata_before_load_raises_runtime_error(fake_oft_modules):
    from wrappers.oft_wrapper import OftWrapper

    wrapper = OftWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial_no_noops")
    with pytest.raises(RuntimeError, match="before load"):
        wrapper.get_metadata()


# --- MailWrapper: unimplemented skeleton, every method raises ---


def test_mail_wrapper_every_method_raises_not_implemented_error():
    from wrappers.mail_wrapper import MailWrapper

    wrapper = MailWrapper(checkpoint="fake/checkpoint", unnorm_key="libero_spatial")
    with pytest.raises(NotImplementedError):
        wrapper.load()
    with pytest.raises(NotImplementedError):
        wrapper.reset()
    with pytest.raises(NotImplementedError):
        wrapper.predict({}, "task")
    with pytest.raises(NotImplementedError):
        wrapper.get_metadata()
    with pytest.raises(NotImplementedError):
        wrapper.get_vram_footprint_gb()
