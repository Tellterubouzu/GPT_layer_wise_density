from typing import List

import torch


def get_transformer_blocks(model: torch.nn.Module) -> List[torch.nn.Module]:
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise ValueError("Unsupported model architecture for layer access")
