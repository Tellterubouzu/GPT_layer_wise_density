import torch

from losses.delta_orth import delta_orth_loss


def test_detach_prev_blocks_grad():
    delta = torch.randn(2, 4, 8, requires_grad=True)
    prev = torch.randn(2, 4, 8, requires_grad=True)

    loss = delta_orth_loss(delta, prev, detach_prev=True)
    loss.backward()

    assert delta.grad is not None
    assert prev.grad is None


def test_no_detach_allows_grad():
    delta = torch.randn(2, 4, 8, requires_grad=True)
    prev = torch.randn(2, 4, 8, requires_grad=True)

    loss = delta_orth_loss(delta, prev, detach_prev=False)
    loss.backward()

    assert delta.grad is not None
    assert prev.grad is not None
