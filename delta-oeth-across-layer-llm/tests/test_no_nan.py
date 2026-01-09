import torch

from losses.delta_orth import delta_orth_loss


def test_no_nan_with_small_norms():
    delta = torch.zeros(2, 4, 8)
    prev = torch.zeros(2, 4, 8)
    loss = delta_orth_loss(delta, prev, eps=1e-6)
    assert torch.isfinite(loss).item()
