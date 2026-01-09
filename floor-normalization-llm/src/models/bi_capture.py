from typing import List


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
