from typing import Iterable, Literal, Optional, Sequence

import torch


TokenReduce = Literal["mean", "sample_k", "pool"]


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    dot = (a * b).sum(dim=-1)
    denom = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    return dot / denom.clamp_min(eps)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def _sample_cosine(
    cos: torch.Tensor, mask: Optional[torch.Tensor], sample_k: int
) -> torch.Tensor:
    if sample_k <= 0:
        return cos.mean()

    bsz, seq_len = cos.shape
    sample_means = []
    for idx in range(bsz):
        if seq_len == 0:
            continue
        if mask is None:
            valid = None
        else:
            valid = torch.nonzero(mask[idx], as_tuple=False).squeeze(-1)
            if valid.numel() == 0:
                continue
        if valid is None:
            k = min(sample_k, seq_len)
            choices = torch.randint(0, seq_len, (k,), device=cos.device)
        else:
            k = min(sample_k, valid.numel())
            choices = valid[torch.randint(0, valid.numel(), (k,), device=cos.device)]
        sample_means.append(cos[idx, choices].mean())

    if not sample_means:
        return torch.tensor(0.0, device=cos.device)
    return torch.stack(sample_means).mean()


def _pool_tokens(
    x_in: torch.Tensor, x_out: torch.Tensor, mask: Optional[torch.Tensor], eps: float
) -> torch.Tensor:
    if x_in.dim() != 3:
        cos = _cosine_similarity(x_in, x_out, eps=eps)
        return cos.mean()

    if mask is None:
        pooled_in = x_in.mean(dim=1)
        pooled_out = x_out.mean(dim=1)
    else:
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_in = (x_in * mask).sum(dim=1) / denom
        pooled_out = (x_out * mask).sum(dim=1) / denom

    cos = _cosine_similarity(pooled_in, pooled_out, eps=eps)
    return cos.mean()


def compute_bi(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    token_reduce: TokenReduce = "mean",
    sample_k: int = 32,
    fp32: bool = True,
    detach_input: bool = False,
    detach_output: bool = False,
) -> torch.Tensor:
    if detach_input:
        x_in = x_in.detach()
    if detach_output:
        x_out = x_out.detach()

    if fp32:
        x_in = x_in.float()
        x_out = x_out.float()

    if x_in.dim() == 2:
        cos = _cosine_similarity(x_in, x_out, eps=eps)
        return 1.0 - cos.mean()
    if x_in.dim() != 3:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {x_in.shape}")

    mask = None
    if attention_mask is not None:
        mask = attention_mask.to(dtype=x_in.dtype)
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)

    if token_reduce == "pool":
        mean_cos = _pool_tokens(x_in, x_out, mask, eps)
        return 1.0 - mean_cos

    cos = _cosine_similarity(x_in, x_out, eps=eps)

    if token_reduce == "sample_k":
        token_mask = None
        if attention_mask is not None:
            token_mask = attention_mask.to(dtype=torch.bool)
        mean_cos = _sample_cosine(cos, token_mask, sample_k)
        return 1.0 - mean_cos

    if token_reduce != "mean":
        raise ValueError(f"Unknown token_reduce: {token_reduce}")

    if attention_mask is None:
        mean_cos = cos.mean()
    else:
        mean_cos = _masked_mean(cos, mask.squeeze(-1))
    return 1.0 - mean_cos


def compute_bi_list(
    hidden_states: Sequence[torch.Tensor],
    layers: Iterable[int],
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    token_reduce: TokenReduce = "mean",
    sample_k: int = 32,
    fp32: bool = True,
    detach_input: bool = False,
    detach_output: bool = False,
) -> torch.Tensor:
    bis = []
    for idx in layers:
        bi = compute_bi(
            hidden_states[idx],
            hidden_states[idx + 1],
            attention_mask=attention_mask,
            eps=eps,
            token_reduce=token_reduce,
            sample_k=sample_k,
            fp32=fp32,
            detach_input=detach_input,
            detach_output=detach_output,
        )
        bis.append(bi)
    if not bis:
        return torch.tensor([], device=hidden_states[0].device)
    return torch.stack(bis)


def bi_floor_loss(
    bi_list: torch.Tensor,
    tau: float,
    mode: str = "hinge",
    beta: float = 20.0,
) -> torch.Tensor:
    if bi_list.numel() == 0:
        return torch.tensor(0.0, device=bi_list.device)

    tau_tensor = torch.tensor(tau, device=bi_list.device, dtype=bi_list.dtype)

    if mode == "hinge":
        return torch.relu(tau_tensor - bi_list).mean()
    if mode == "softmin":
        bi_min = -torch.logsumexp(-beta * bi_list, dim=0) / beta
        return torch.relu(tau_tensor - bi_min)
    raise ValueError(f"Unknown BI-Floor mode: {mode}")
