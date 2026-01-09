import torch


def hinge_floor(values: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.clamp(tau - values, min=0.0)


def schedule_value(step: int, warmup_steps: int, ramp_steps: int, max_value: float) -> float:
    if max_value <= 0.0:
        return 0.0
    if step < warmup_steps:
        return 0.0
    if ramp_steps <= 0:
        return max_value
    progress = (step - warmup_steps) / float(ramp_steps)
    return max_value * min(1.0, max(0.0, progress))
