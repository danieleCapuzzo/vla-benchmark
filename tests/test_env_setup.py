"""
Standalone environment verification for the prerequisites documented in README.md:
torch+CUDA, causal-conv1d (with its compiled `causal_conv1d_cuda` extension), and
mamba-ssm (with its compiled `selective_scan_cuda` extension).

These three are deliberately NOT declared in pyproject.toml -- they're CUDA/GPU
specific and must be installed by hand before `pip install .` (see README.md). This
file is the thing you run right after that manual install to confirm it actually
worked, before trusting the rest of the test suite.

Usage:
    python tests/test_env_setup.py      # standalone, verbose, human-readable
    pytest tests/test_env_setup.py -v   # as part of the normal test suite
"""

from __future__ import annotations

import sys

import pytest

_SEE_README = "See README.md for install instructions."


def _fail(msg: str) -> None:
    pytest.fail(f"{msg}\n{_SEE_README}")


# --------------------------------------------------------------------------------
# torch + CUDA
# --------------------------------------------------------------------------------


def test_torch_importable_with_cuda_build() -> None:
    try:
        import torch
    except ImportError as e:
        _fail(f"torch is not installed ({e}).")
        return
    if torch.version.cuda is None:
        _fail(
            "torch is installed but is a CPU-only build (torch.version.cuda is None). "
            "Install a CUDA build of torch matching your driver's CUDA version."
        )


def test_cuda_device_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        _fail(
            "torch.cuda.is_available() is False. Either no GPU is visible to this "
            "process, or the installed torch CUDA build doesn't match the driver / "
            "GPU compute capability (check `nvidia-smi` and torch's CUDA version)."
        )
    name = torch.cuda.get_device_name(0)
    assert isinstance(name, str) and name


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


# The remaining tests exercise live CUDA kernels and need an actual device. Like
# tests/test_mail_env_smoke.py, they're skipped (not failed) when none is visible, so
# `pytest` stays green on CPU-only machines. `test_cuda_device_available` above is the
# one that fails loudly if you expected a GPU but torch can't see it.
requires_cuda = pytest.mark.skipif(
    not _cuda_available(),
    reason="requires a CUDA device",
)


@requires_cuda
def test_cuda_tensor_matmul_roundtrip() -> None:
    """Exercises the CUDA runtime itself, not just `is_available()`."""
    torch = pytest.importorskip("torch")
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    assert c.shape == (64, 64)
    assert torch.isfinite(c).all()


# --------------------------------------------------------------------------------
# causal-conv1d
# --------------------------------------------------------------------------------


def test_causal_conv1d_importable() -> None:
    try:
        import causal_conv1d  # noqa: F401
    except ImportError as e:
        _fail(
            f"causal-conv1d is not installed ({e}). It must be installed for your "
            "torch/CUDA combination -- a matching prebuilt wheel, or compiled from "
            "source if none is available."
        )


def test_causal_conv1d_cuda_extension_loads() -> None:
    """
    The pure-Python `causal_conv1d` package is useless without its compiled CUDA
    extension. If the extension failed to build/load, causal_conv1d_fn below will
    silently (or loudly) fall back or error -- so import the compiled `.so` directly.
    """
    pytest.importorskip("causal_conv1d")
    try:
        import causal_conv1d_cuda  # noqa: F401
    except ImportError as e:
        _fail(
            f"causal_conv1d is installed but its compiled CUDA extension "
            f"(causal_conv1d_cuda) failed to import ({e}). This usually means the "
            "prebuilt wheel didn't match your torch/CUDA ABI and it needs to be "
            "compiled from source instead."
        )


@requires_cuda
def test_causal_conv1d_fn_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("causal_conv1d")
    from causal_conv1d import causal_conv1d_fn

    batch, channels, seqlen, width = 2, 16, 32, 4
    x = torch.randn(batch, channels, seqlen, device="cuda", requires_grad=True)
    weight = torch.randn(channels, width, device="cuda", requires_grad=True)
    bias = torch.randn(channels, device="cuda", requires_grad=True)

    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    assert out.shape == (batch, channels, seqlen)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


# --------------------------------------------------------------------------------
# mamba-ssm
# --------------------------------------------------------------------------------


def test_mamba_ssm_importable() -> None:
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        _fail(
            f"mamba-ssm is not installed ({e}). It must be installed for your "
            "torch/CUDA combination -- a matching prebuilt wheel, or compiled from "
            "source if none is available."
        )


def test_selective_scan_cuda_extension_loads() -> None:
    """mamba-ssm's actual selective-scan CUDA kernel, compiled separately from the pure-Python package."""
    pytest.importorskip("mamba_ssm")
    try:
        import selective_scan_cuda  # noqa: F401
    except ImportError as e:
        _fail(
            f"mamba-ssm is installed but its compiled CUDA kernel "
            f"(selective_scan_cuda) failed to import ({e}). This usually means the "
            "prebuilt wheel didn't match your torch/CUDA ABI and it needs to be "
            "compiled from source instead."
        )


@requires_cuda
def test_mamba_block_forward_and_backward() -> None:
    """End-to-end: a real Mamba block, on GPU, exercising both compiled kernels together."""
    torch = pytest.importorskip("torch")
    mamba_ssm = pytest.importorskip("mamba_ssm")

    Mamba = mamba_ssm.Mamba
    batch, seqlen, d_model = 2, 32, 64

    model = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2).to("cuda")
    x = torch.randn(batch, seqlen, d_model, device="cuda", requires_grad=True)

    out = model(x)
    assert out.shape == (batch, seqlen, d_model)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grad_norms and any(g > 0 for g in grad_norms)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"] + sys.argv[1:]))
