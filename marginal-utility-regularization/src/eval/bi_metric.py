from typing import Dict, List

import torch
from torch.utils.data import DataLoader


def compute_bi_metric(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device, eps: float = 1e-6
) -> Dict[str, List[float]]:
    model.eval()
    accum: List[float] = []
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

            if not accum:
                accum = [0.0 for _ in range(num_layers)]
                counts = [0 for _ in range(num_layers)]

            for idx in range(num_layers):
                h_in = hidden_states[idx]
                h_out = hidden_states[idx + 1]
                dot = (h_in * h_out).sum(dim=-1)
                norm_in = torch.linalg.norm(h_in, dim=-1)
                norm_out = torch.linalg.norm(h_out, dim=-1)
                cos = dot / (norm_in * norm_out).clamp_min(eps)
                bi = 1.0 - cos
                accum[idx] += bi.mean().item()
                counts[idx] += 1

    model.train()
    return {"bi": [accum[i] / max(1, counts[i]) for i in range(len(accum))]}
