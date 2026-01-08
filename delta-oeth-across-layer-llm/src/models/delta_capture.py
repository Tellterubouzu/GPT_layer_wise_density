from typing import Iterable, List, Sequence

import torch


def hidden_states_to_deltas(hidden_states: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    if len(hidden_states) < 2:
        raise ValueError("hidden_states must contain at least two elements")
    deltas = []
    for idx in range(len(hidden_states) - 1):
        deltas.append(hidden_states[idx + 1] - hidden_states[idx])
    return deltas


def resolve_layer_indices(total_layers: int, mid_start: float, mid_end: float) -> List[int]:
    if total_layers <= 0:
        return []

    def _resolve(value: float, default: int) -> int:
        if 0.0 < value < 1.0:
            return int(round(total_layers * value))
        if value >= 1.0:
            return int(value)
        return default

    start = _resolve(mid_start, 0)
    end = _resolve(mid_end, total_layers)
    start = max(0, min(total_layers - 1, start))
    end = max(start + 1, min(total_layers, end))
    return list(range(start, end))


def adjacent_pairs(indices: Iterable[int]) -> List[tuple]:
    sorted_idx = sorted(indices)
    return [(idx, idx - 1) for idx in sorted_idx if idx - 1 in indices]
