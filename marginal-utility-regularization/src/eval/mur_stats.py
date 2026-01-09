from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from models.hidden_state_utils import hidden_states_to_deltas, resolve_layer_indices
from mur.utility import compute_metric_values, sample_token_values, summarize_values


def compute_mur_stats(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metric: str = "cos",
    mid_start: float = 0.0,
    mid_end: float = 1.0,
    tau: float = 0.0,
    sample_k: int = 64,
    eps: float = 1e-6,
    fp32_dot: bool = True,
) -> Dict[str, List[float]]:
    model.eval()

    layer_indices: Optional[List[int]] = None
    values_store: List[List[torch.Tensor]] = []
    delta_norms: List[List[torch.Tensor]] = []
    grad_norms: List[List[torch.Tensor]] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Hidden states are required for MUR stats")
        if layer_indices is None:
            num_layers = len(hidden_states) - 1
            layer_indices = resolve_layer_indices(num_layers, mid_start, mid_end)
            values_store = [[] for _ in layer_indices]
            delta_norms = [[] for _ in layer_indices]
            grad_norms = [[] for _ in layer_indices]

        h_out = [hidden_states[idx + 1] for idx in layer_indices]
        grads = torch.autograd.grad(outputs.loss, h_out, retain_graph=False, create_graph=False, allow_unused=True)
        deltas = hidden_states_to_deltas(hidden_states)

        for pos, (layer_idx, grad) in enumerate(zip(layer_indices, grads)):
            if grad is None:
                continue
            delta = deltas[layer_idx]
            values = compute_metric_values(grad, delta, metric, eps, fp32_dot)
            values_store[pos].append(sample_token_values(values, sample_k).detach().cpu())
            delta_norm = torch.linalg.norm(delta, dim=-1)
            grad_norm = torch.linalg.norm(grad, dim=-1)
            delta_norms[pos].append(sample_token_values(delta_norm, sample_k).detach().cpu())
            grad_norms[pos].append(sample_token_values(grad_norm, sample_k).detach().cpu())

    if layer_indices is None:
        return {
            "metric": metric,
            "layers": [],
            "mean": [],
            "p10": [],
            "p50": [],
            "p90": [],
            "frac_below_tau": [],
            "mean_delta_norm": [],
            "mean_grad_norm": [],
        }

    stats: Dict[str, List[float]] = {
        "metric": metric,
        "layers": layer_indices,
        "mean": [],
        "p10": [],
        "p50": [],
        "p90": [],
        "frac_below_tau": [],
        "mean_delta_norm": [],
        "mean_grad_norm": [],
    }

    for pos in range(len(layer_indices)):
        if values_store[pos]:
            concat = torch.cat(values_store[pos])
        else:
            concat = torch.tensor([])
        summary = summarize_values(concat, tau)
        stats["mean"].append(summary["mean"])
        stats["p10"].append(summary["p10"])
        stats["p50"].append(summary["p50"])
        stats["p90"].append(summary["p90"])
        stats["frac_below_tau"].append(summary["frac_below_tau"])

        if delta_norms[pos]:
            delta_concat = torch.cat(delta_norms[pos])
            stats["mean_delta_norm"].append(delta_concat.mean().item())
        else:
            stats["mean_delta_norm"].append(0.0)

        if grad_norms[pos]:
            grad_concat = torch.cat(grad_norms[pos])
            stats["mean_grad_norm"].append(grad_concat.mean().item())
        else:
            stats["mean_grad_norm"].append(0.0)

    model.train()
    return stats
