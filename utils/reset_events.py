from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ResetResult:
    env_ids: torch.Tensor
    reset_type: str
    terrain_ids: torch.Tensor
    start_xy: torch.Tensor
    goal_xy: Optional[torch.Tensor]
    root_local_pos: torch.Tensor
    done_mask: Optional[torch.Tensor] = None

