from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from losses.delta_orth import cosine_stats
from models.delta_capture import hidden_states_to_deltas


def compute_delta_stats(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    token_reduce: str = "mean",
    sample_k: int = 32,
    eps: float = 1e-6,
) -> Dict[str, List[float]]:
    model.eval()
    cos_values: List[List[torch.Tensor]] = []
    norm_values: List[List[torch.Tensor]] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            deltas = hidden_states_to_deltas(outputs.hidden_states)
            num_layers = len(deltas)

            if not cos_values:
                cos_values = [[] for _ in range(num_layers - 1)]
                norm_values = [[] for _ in range(num_layers)]

            for layer_idx in range(num_layers):
                norms = torch.linalg.norm(deltas[layer_idx], dim=-1).flatten().cpu()
                norm_values[layer_idx].append(norms)

            for layer_idx in range(1, num_layers):
                cos = cosine_stats(
                    deltas[layer_idx],
                    deltas[layer_idx - 1],
                    eps=eps,
                    reduce_tokens=token_reduce,
                    sample_k=sample_k,
                )
                cos_values[layer_idx - 1].append(cos.detach().cpu())

    stats: Dict[str, List[float]] = {
        "mean_cos": [],
        "p90_cos": [],
        "p99_cos": [],
        "mean_delta_norm": [],
    }

    for layer_idx, layer_cos in enumerate(cos_values):
        concat = torch.cat(layer_cos)
        stats["mean_cos"].append(concat.mean().item())
        stats["p90_cos"].append(torch.quantile(concat, 0.9).item())
        stats["p99_cos"].append(torch.quantile(concat, 0.99).item())

    for layer_idx, layer_norm in enumerate(norm_values):
        concat = torch.cat(layer_norm)
        stats["mean_delta_norm"].append(concat.mean().item())

    model.train()
    return stats
