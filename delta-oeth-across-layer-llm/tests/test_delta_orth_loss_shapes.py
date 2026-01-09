import torch

from losses.delta_orth import delta_orth_loss


def test_delta_orth_loss_shapes_3d():
    delta = torch.randn(2, 4, 8)
    prev = torch.randn(2, 4, 8)
    loss = delta_orth_loss(delta, prev, reduce_tokens="mean")
    assert loss.dim() == 0


def test_delta_orth_loss_shapes_2d():
    delta = torch.randn(3, 8)
    prev = torch.randn(3, 8)
    loss = delta_orth_loss(delta, prev, reduce_tokens="pool")
    assert loss.dim() == 0
