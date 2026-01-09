from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from eval.ppl import compute_ppl
from models.block_access import get_transformer_blocks


def _make_skip_hook(layer_idx: int):
    def hook(module, inputs, outputs):
        hidden = inputs[0]
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    return hook


def compute_layer_drop(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> Dict[str, List[float]]:
    model.eval()
    blocks = get_transformer_blocks(model)

    base = compute_ppl(model, dataloader, device)
    base_ppl = base["ppl"]

    drop_ppl: List[float] = []
    for idx, block in enumerate(blocks):
        hook = block.register_forward_hook(_make_skip_hook(idx))
        result = compute_ppl(model, dataloader, device)
        drop_ppl.append(result["ppl"] - base_ppl)
        hook.remove()

    model.train()
    return {"delta_ppl": drop_ppl, "base_ppl": base_ppl}
