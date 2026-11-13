from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .encoders import CommandConditionEncoder, CommandConditionEncoderConfig, build_mlp


@dataclass
class DDPMCommandDenoiserConfig:
    encoder: CommandConditionEncoderConfig
    horizon: int = 10
    command_dim: int = 4
    timestep_dim: int = 64
    hidden_dims: Tuple[int, int, int] = (512, 512, 512)


class DDPMCommandDenoiser(nn.Module):
    def __init__(self, cfg: DDPMCommandDenoiserConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = CommandConditionEncoder(cfg.encoder)
        self.timestep_mlp = build_mlp(int(cfg.timestep_dim), (int(cfg.timestep_dim) * 2,), int(cfg.timestep_dim))
        input_dim = int(cfg.horizon) * int(cfg.command_dim) + int(cfg.encoder.output_dim) + int(cfg.timestep_dim)
        self.denoiser = build_mlp(input_dim, cfg.hidden_dims, int(cfg.horizon) * int(cfg.command_dim))

    def forward(self, batch: dict[str, torch.Tensor], noisy_commands: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        cond = self.encode_condition(batch)
        return self.denoise_from_condition(cond, noisy_commands, timesteps)

    def encode_condition(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encoder(batch)

    def denoise_from_condition(
        self,
        condition: torch.Tensor,
        noisy_commands: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.timestep_mlp(sinusoidal_timestep_embedding(timesteps, int(self.cfg.timestep_dim)))
        x = torch.cat([noisy_commands.flatten(start_dim=1), condition, time_features], dim=-1)
        pred_noise = self.denoiser(x)
        return pred_noise.view_as(noisy_commands)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    device = timesteps.device
    exponent = -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
    args = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding
