from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
