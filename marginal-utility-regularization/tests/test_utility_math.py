import torch

from mur.utility import compute_metric_values, utility_cos


def test_utility_cos_range() -> None:
    g = torch.randn(2, 3, 4)
    d = torch.randn(2, 3, 4)
    values = utility_cos(g, d)
    assert values.max().item() <= 1.0001
    assert values.min().item() >= -1.0001


def test_metric_no_nan_with_zeros() -> None:
    g = torch.zeros(2, 3, 4)
    d = torch.zeros(2, 3, 4)
    values = compute_metric_values(g, d, metric="cos", eps=1e-6, fp32_dot=True)
    assert not torch.isnan(values).any()
