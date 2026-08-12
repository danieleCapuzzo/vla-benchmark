"""
Smoke test for the mamba-ssm + CUDA kernels.
Approach: intentionally exercises the real mamba_ssm package and real CUDA kernels (selective_scan_cuda, causal_conv1d) 
to verify the environment build itself
"""

import pytest

torch = pytest.importorskip("torch")
mamba_ssm = pytest.importorskip("mamba_ssm")

Mamba = mamba_ssm.Mamba

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mail-env smoke test requires a CUDA device (mamba_ssm kernels are CUDA-only)",
)


class _TinyMambaModel(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.head = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mamba(x))


def test_mail_env_reports_expected_cuda_device():
    assert torch.version.cuda is not None
    device_name = torch.cuda.get_device_name(0)
    assert isinstance(device_name, str) and device_name


def test_mail_env_mamba_forward_and_backward_step():
    device = "cuda"
    batch_size, seq_len, d_model = 4, 32, 64

    model = _TinyMambaModel(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    inputs = torch.randn(batch_size, seq_len, d_model, device=device)
    targets = torch.randn(batch_size, seq_len, d_model, device=device)

    optimizer.zero_grad()
    outputs = model(inputs)
    assert outputs.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(outputs).all()

    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert grad_norms, "no gradients were populated -- backward pass did not reach mamba params"
    assert any(g > 0 for g in grad_norms)
