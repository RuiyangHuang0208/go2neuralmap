from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .encoders import build_mlp


@dataclass
class RiskClassifierConfig:
    condition_dim: int = 256
    horizon: int = 10
    command_dim: int = 4
    hidden_dims: Tuple[int, int] = (256, 256)


class RiskClassifier(nn.Module):
    def __init__(self, cfg: RiskClassifierConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = int(cfg.condition_dim) + int(cfg.horizon) * int(cfg.command_dim)
        self.net = build_mlp(input_dim, cfg.hidden_dims, 1)

    def forward(self, condition: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.dim() == 3:
            x = torch.cat([condition, candidates.flatten(start_dim=1)], dim=-1)
            return self.net(x).squeeze(-1)
        if candidates.dim() != 4:
            raise ValueError(f"candidates must have shape [B, H, D] or [B, K, H, D], got {tuple(candidates.shape)}")
        batch_size, num_candidates = candidates.shape[:2]
        condition_expanded = condition.unsqueeze(1).expand(-1, num_candidates, -1)
        x = torch.cat(
            [
                condition_expanded.reshape(batch_size * num_candidates, -1),
                candidates.reshape(batch_size * num_candidates, -1),
            ],
            dim=-1,
        )
        return self.net(x).view(batch_size, num_candidates)
