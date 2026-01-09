import os
from typing import Optional

import torch


def save_model_state(model: torch.nn.Module, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_state.pt")
    torch.save(model.state_dict(), path)
    return path


def load_model_state(model: torch.nn.Module, run_dir: str, map_location: Optional[str] = None) -> bool:
    path = os.path.join(run_dir, "model_state.pt")
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return True
