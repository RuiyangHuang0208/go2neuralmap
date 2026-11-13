from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn


@dataclass
class CommandConditionEncoderConfig:
    crop_size: int = 8
    latent_dim: int = 64
    map_channels: int = 260
    cnn_channels: Tuple[int, int] = (64, 128)
    vector_hidden_dims: Tuple[int, int] = (256, 256)
    output_dim: int = 256


class CommandConditionEncoder(nn.Module):
    def __init__(self, cfg: CommandConditionEncoderConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if cfg.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        conv_layers = []
        in_channels = int(cfg.map_channels)
        for out_channels in cfg.cnn_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, int(out_channels), kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(8, int(out_channels)), num_channels=int(out_channels)),
                nn.SiLU(),
            ])
            in_channels = int(out_channels)
        conv_layers.append(nn.Flatten())
        self.map_encoder = nn.Sequential(*conv_layers)

        map_feature_dim = int(in_channels) * int(cfg.crop_size) * int(cfg.crop_size)
        vector_input_dim = 2 + 2 + 4 * int(cfg.latent_dim)
        self.vector_encoder = build_mlp(vector_input_dim, cfg.vector_hidden_dims, int(cfg.output_dim))
        self.fusion = build_mlp(map_feature_dim + int(cfg.output_dim), (int(cfg.output_dim),), int(cfg.output_dim))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        map_input = torch.cat([batch["map_crop"], batch["uncertainty_crop"]], dim=1)
        map_features = self.map_encoder(map_input)
        vector_input = torch.cat(
            [
                batch["goal_delta"],
                batch["current_heading"],
                batch["current_latent"],
            ],
            dim=-1,
        )
        vector_features = self.vector_encoder(vector_input)
        return self.fusion(torch.cat([map_features, vector_features], dim=-1))


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, int(hidden_dim)), nn.SiLU()])
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)
