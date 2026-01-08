from typing import Dict, List

import torch
from torch.utils.data import DataLoader


def _linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x.T @ y).pow(2).sum()
    norm_x = (x.T @ x).pow(2).sum()
    norm_y = (y.T @ y).pow(2).sum()
    return hsic / (norm_x * norm_y + eps).sqrt()


def compute_adjacent_cka(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2048,
) -> Dict[str, List[float]]:
    model.eval()
    values: List[float] = []
    counts: List[int] = []

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
            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states) - 1
            if not values:
                values = [0.0 for _ in range(num_layers)]
                counts = [0 for _ in range(num_layers)]

            for idx in range(num_layers):
                x = hidden_states[idx].reshape(-1, hidden_states[idx].shape[-1])
                y = hidden_states[idx + 1].reshape(-1, hidden_states[idx + 1].shape[-1])
                if x.size(0) > max_samples:
                    indices = torch.randperm(x.size(0), device=x.device)[:max_samples]
                    x = x[indices]
                    y = y[indices]
                cka = _linear_cka(x, y)
                values[idx] += cka.item()
                counts[idx] += 1

    model.train()
    return {"cka": [values[i] / max(1, counts[i]) for i in range(len(values))]}
