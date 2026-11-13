from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class RLNavCommandConfig:
    max_forward_speed: float = 1.5
    max_backward_speed: float = 0.5
    max_lateral_speed: float = 0.5
    max_yaw_rate: float = 1.0
    deadband: float = 0.0


def scale_rl_nav_actions(
    actions: torch.Tensor,
    cfg: RLNavCommandConfig,
    *,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if actions.dim() != 2 or actions.shape[-1] != 3:
        raise ValueError(f"actions must have shape [num_envs, 3], got {tuple(actions.shape)}")

    normalized = torch.clamp(actions, min=-1.0, max=1.0)
    if float(cfg.deadband) > 0.0:
        normalized = torch.where(
            normalized.abs() < float(cfg.deadband),
            torch.zeros_like(normalized),
            normalized,
        )

    commands = torch.zeros_like(normalized)
    commands[:, 0] = torch.where(
        normalized[:, 0] >= 0.0,
        normalized[:, 0] * float(cfg.max_forward_speed),
        normalized[:, 0] * float(cfg.max_backward_speed),
    )
    commands[:, 1] = normalized[:, 1] * float(cfg.max_lateral_speed)
    commands[:, 2] = normalized[:, 2] * float(cfg.max_yaw_rate)

    if not return_info:
        return commands

    return commands, {
        "normalized_action": normalized,
        "command_speed": torch.norm(commands[:, :2], dim=-1),
        "yaw_rate_abs": commands[:, 2].abs(),
    }
