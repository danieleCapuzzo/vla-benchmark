"""
Unit tests for reachable validation branches in wrappers/: bad quant, and
get_metadata()-before-load() for all three wrappers.

OpenVlaWrapper's and MailWrapper's validation raises before any torch.cuda call, so they
need no mocking -- both modules import cleanly in .oft-env. OftWrapper's module-level
imports require the same sys.modules faking as tests/test_oft_wrapper.py, so this file
reuses that fixture.
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


# # --- MailWrapper: validation raises before any checkpoint load ---
def test_mail_wrapper_load_bad_quant_raises_value_error():
    from wrappers.mail_wrapper import MailWrapper

    wrapper = MailWrapper(checkpoint="fake/checkpoint.pt", unnorm_key="")
    with pytest.raises(ValueError, match="quantized loading path"):
        wrapper.load(quant="8bit")


def test_mail_wrapper_get_metadata_before_load_raises_runtime_error():
    from wrappers.mail_wrapper import MailWrapper

    wrapper = MailWrapper(checkpoint="fake/checkpoint.pt", unnorm_key="")
    with pytest.raises(RuntimeError, match="before load"):
        wrapper.get_metadata()


# --- RoboMambaWrapper: validation raises before any checkpoint/trunk load ---
def test_robomamba_wrapper_load_bad_quant_raises_value_error():
    from wrappers.robomamba_wrapper import RoboMambaWrapper

    wrapper = RoboMambaWrapper(checkpoint="fake/checkpoint.pt", unnorm_key="")
    with pytest.raises(ValueError, match="quantized loading path"):
        wrapper.load(quant="8bit")


def test_robomamba_wrapper_get_metadata_before_load_raises_runtime_error():
    from wrappers.robomamba_wrapper import RoboMambaWrapper

    wrapper = RoboMambaWrapper(checkpoint="fake/checkpoint.pt", unnorm_key="")
    with pytest.raises(RuntimeError, match="before load"):
        wrapper.get_metadata()
