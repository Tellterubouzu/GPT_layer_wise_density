from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from eval.ppl import compute_ppl


def _get_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported model architecture for layer drop")


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
    blocks = _get_blocks(model)

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
