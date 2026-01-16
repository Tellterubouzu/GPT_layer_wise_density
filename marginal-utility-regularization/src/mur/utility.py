from typing import Dict, Tuple

import torch


def _maybe_fp32(g: torch.Tensor, d: torch.Tensor, fp32_dot: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if fp32_dot:
        return g.float(), d.float()
    return g, d


def utility_raw(g: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return -(g * d).sum(dim=-1)


def utility_cos(g: torch.Tensor, d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (g * d).sum(dim=-1)
    ng = torch.linalg.norm(g, dim=-1)
    nd = torch.linalg.norm(d, dim=-1)
    return -(dot / (ng * nd).clamp_min(eps))


def utility_proj(g: torch.Tensor, d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (g * d).sum(dim=-1)
    ng2 = (g * g).sum(dim=-1)
    return -(dot / ng2.clamp_min(eps))


def compute_metric_values(
    g: torch.Tensor,
    d: torch.Tensor,
    metric: str,
    eps: float,
    fp32_dot: bool,
) -> torch.Tensor:
    g, d = _maybe_fp32(g, d, fp32_dot)
    if metric == "cos":
        return utility_cos(g, d, eps=eps)
    if metric == "proj":
        return utility_proj(g, d, eps=eps)
    if metric == "raw":
        return utility_raw(g, d)
    raise ValueError(f"Unknown MUR metric: {metric}")


def reduce_token_values(values: torch.Tensor, reduce: str, sample_k: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    if reduce == "mean":
        return flat.mean()
    if reduce == "sample_k":
        if sample_k <= 0:
            return flat.mean()
        k = min(sample_k, flat.numel())
        idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
        return flat.index_select(0, idx).mean()
    raise ValueError(f"Unknown token reduction: {reduce}")


def sample_token_values(values: torch.Tensor, sample_k: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return flat
    if sample_k <= 0 or sample_k >= flat.numel():
        return flat
    idx = torch.randint(0, flat.numel(), (sample_k,), device=flat.device)
    return flat.index_select(0, idx)


def summarize_values(values: torch.Tensor, tau: float) -> Dict[str, float]:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "frac_below_tau": 0.0}
    flat = flat.float()
    return {
        "mean": flat.mean().item(),
        "p10": torch.quantile(flat, 0.1).item(),
        "p50": torch.quantile(flat, 0.5).item(),
        "p90": torch.quantile(flat, 0.9).item(),
        "frac_below_tau": (flat < tau).float().mean().item(),
    }
