from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .encoders import CommandConditionEncoder, CommandConditionEncoderConfig, build_mlp


@dataclass
class BaselineCommandPolicyConfig:
    encoder: CommandConditionEncoderConfig
    horizon: int = 10
    command_dim: int = 4
    head_hidden_dims: Tuple[int, int] = (256, 256)


class BaselineCommandPolicy(nn.Module):
    def __init__(self, cfg: BaselineCommandPolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CommandConditionEncoder(cfg.encoder)
        self.head = build_mlp(
            int(cfg.encoder.output_dim),
            cfg.head_hidden_dims,
            int(cfg.horizon) * int(cfg.command_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.encoder(batch)
        pred = self.head(cond)
        return pred.view(-1, int(self.cfg.horizon), int(self.cfg.command_dim))
