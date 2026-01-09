import math
from typing import Dict

import torch
from torch.utils.data import DataLoader


def compute_ppl(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    mean_loss = total_loss / max(1, total_tokens)
    return {"loss": mean_loss, "ppl": math.exp(mean_loss)}
