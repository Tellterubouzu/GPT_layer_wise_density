from typing import Literal, Tuple

import torch


TokenReduce = Literal["mean", "sample_k", "pool"]


def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {x.shape}")
    return x.reshape(-1, x.size(-1))


def _sample_tokens(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.dim() != 3:
        return x
    bsz, seq_len, dim = x.shape
    if seq_len <= k:
        return x.reshape(-1, dim)
    indices = torch.randint(0, seq_len, (bsz, k), device=x.device)
    batch_idx = torch.arange(bsz, device=x.device).unsqueeze(-1)
    sampled = x[batch_idx, indices]
    return sampled.reshape(-1, dim)


def _pool_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    return x.mean(dim=1)


def _reduce_tokens(delta: torch.Tensor, prev_delta: torch.Tensor, reduce_tokens: TokenReduce, sample_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if reduce_tokens == "mean":
        return _flatten_tokens(delta), _flatten_tokens(prev_delta)
    if reduce_tokens == "sample_k":
        return _sample_tokens(delta, sample_k), _sample_tokens(prev_delta, sample_k)
    if reduce_tokens == "pool":
        return _pool_tokens(delta), _pool_tokens(prev_delta)
    raise ValueError(f"Unknown reduce_tokens: {reduce_tokens}")


def delta_orth_loss(
    delta: torch.Tensor,
    prev_delta: torch.Tensor,
    eps: float = 1e-6,
    detach_prev: bool = False,
    reduce_tokens: TokenReduce = "mean",
    sample_k: int = 32,
) -> torch.Tensor:
    if detach_prev:
        prev_delta = prev_delta.detach()

    delta_red, prev_red = _reduce_tokens(delta, prev_delta, reduce_tokens, sample_k)

    dot = (delta_red * prev_red).sum(dim=-1)
    norm_delta = torch.linalg.norm(delta_red, dim=-1)
    norm_prev = torch.linalg.norm(prev_red, dim=-1)
    denom = (norm_delta * norm_prev).clamp_min(eps)
    cos = dot / denom
    return (cos ** 2).mean()


def cosine_stats(
    delta: torch.Tensor,
    prev_delta: torch.Tensor,
    eps: float = 1e-6,
    reduce_tokens: TokenReduce = "mean",
    sample_k: int = 32,
) -> torch.Tensor:
    delta_red, prev_red = _reduce_tokens(delta, prev_delta, reduce_tokens, sample_k)
    dot = (delta_red * prev_red).sum(dim=-1)
    norm_delta = torch.linalg.norm(delta_red, dim=-1)
    norm_prev = torch.linalg.norm(prev_red, dim=-1)
    denom = (norm_delta * norm_prev).clamp_min(eps)
    cos = dot / denom
    return cos
