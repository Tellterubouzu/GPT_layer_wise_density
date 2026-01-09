import torch

from losses.bi_floor import compute_bi


def test_no_nan_with_zeros():
    x_in = torch.zeros(2, 4, 8)
    x_out = torch.zeros(2, 4, 8)
    bi = compute_bi(x_in, x_out, eps=1e-6)
    assert torch.isfinite(bi)


def test_masked_mean_ignores_padding():
    x_in = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    x_out = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[1, 0]])
    bi = compute_bi(x_in, x_out, attention_mask=mask, token_reduce="mean")
    assert torch.isclose(bi, torch.tensor(0.0))


def test_detach_input_blocks_grad():
    x_in = torch.randn(2, 3, 4, requires_grad=True)
    x_out = torch.randn(2, 3, 4, requires_grad=True)
    bi = compute_bi(x_in, x_out, detach_input=True)
    bi.backward()
    assert x_in.grad is None
    assert x_out.grad is not None
