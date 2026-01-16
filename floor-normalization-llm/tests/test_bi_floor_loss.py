import torch

from losses.bi_floor import bi_floor_loss


def test_hinge_loss():
    bi_list = torch.tensor([0.1, 0.4])
    loss = bi_floor_loss(bi_list, tau=0.3, mode="hinge")
    expected = (0.3 - 0.1 + 0.0) / 2.0
    assert torch.isclose(loss, torch.tensor(expected))


def test_softmin_loss():
    bi_list = torch.tensor([0.2, 0.5])
    beta = 1.0
    loss = bi_floor_loss(bi_list, tau=0.3, mode="softmin", beta=beta)
    bi_min = -torch.logsumexp(-beta * bi_list, dim=0) / beta
    expected = torch.relu(torch.tensor(0.3) - bi_min)
    assert torch.isclose(loss, expected)
