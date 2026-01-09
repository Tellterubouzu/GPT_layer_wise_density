from typing import Dict, Iterable, List

import torch


def compute_layer_multipliers(layer_indices: Iterable[int], layer_values: Iterable[float], tau: float, alpha: float) -> Dict[int, float]:
    multipliers: Dict[int, float] = {}
    for idx, value in zip(layer_indices, layer_values):
        penalty = max(0.0, tau - float(value))
        multipliers[idx] = 1.0 + alpha * penalty
    return multipliers


def scale_layer_grads(blocks: List[torch.nn.Module], multipliers: Dict[int, float]) -> None:
    for idx, block in enumerate(blocks):
        scale = multipliers.get(idx, 1.0)
        if scale == 1.0:
            continue
        for param in block.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
