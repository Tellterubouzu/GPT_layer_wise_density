from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from losses.bi_floor import compute_bi_list


def compute_bi_metric(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    token_reduce: str = "mean",
    sample_k: int = 32,
    eps: float = 1e-6,
    fp32: bool = True,
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

            bi_list = compute_bi_list(
                hidden_states,
                range(num_layers),
                attention_mask=attention_mask,
                eps=eps,
                token_reduce=token_reduce,
                sample_k=sample_k,
                fp32=fp32,
            )

            for idx in range(num_layers):
                accum[idx] += bi_list[idx].item()
                counts[idx] += 1

    model.train()
    return {"bi": [accum[i] / max(1, counts[i]) for i in range(len(accum))]}
