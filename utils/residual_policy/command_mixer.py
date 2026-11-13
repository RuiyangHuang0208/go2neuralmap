from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class CommandMixerConfig:
    residual_horizon: int = 2
    mix_mode: str = "additive"
    residual_vxy_scale: float = 0.3
    residual_heading_scale: float = 0.4
    residual_vel_clip: Optional[float] = None
    residual_heading_clip: Optional[float] = None
    min_velocity_scale: float = 0.3
    max_velocity_scale: float = 1.2
    max_forward_speed: float = 1.5
    max_lateral_speed: float = 0.5


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def mix_residual_commands(
    base_commands: torch.Tensor,
    residual_actions: torch.Tensor,
    cfg: CommandMixerConfig,
    *,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compose frozen diffusion commands with normalized residual actions.

    Args:
        base_commands: Tensor with shape [num_envs, residual_horizon, 3].
        residual_actions: Tensor with shape [num_envs, residual_horizon * 3]
            or [num_envs, residual_horizon, 3].
        cfg: Residual command scaling and clipping parameters.
    """
    if base_commands.dim() != 3 or base_commands.shape[-1] != 3:
        raise ValueError(f"base_commands must have shape [N, H, 3], got {tuple(base_commands.shape)}")
    residual_horizon = int(cfg.residual_horizon)
    if residual_horizon <= 0:
        raise ValueError("residual_horizon must be positive")
    if base_commands.shape[1] != residual_horizon:
        raise ValueError(
            f"base_commands horizon must be {residual_horizon}, got {base_commands.shape[1]}"
        )

    if residual_actions.dim() == 2:
        expected_actions = residual_horizon * 3
        if residual_actions.shape != (base_commands.shape[0], expected_actions):
            raise ValueError(
                "flat residual_actions must have shape "
                f"({base_commands.shape[0]}, {expected_actions}), got {tuple(residual_actions.shape)}"
            )
        residual = residual_actions.view(base_commands.shape[0], residual_horizon, 3)
    elif residual_actions.dim() == 3:
        if residual_actions.shape != base_commands.shape:
            raise ValueError(
                f"residual_actions must have shape {tuple(base_commands.shape)}, "
                f"got {tuple(residual_actions.shape)}"
            )
        residual = residual_actions
    else:
        raise ValueError(f"residual_actions must be rank 2 or 3, got rank {residual_actions.dim()}")

    residual = residual.to(device=base_commands.device, dtype=base_commands.dtype)
    mixed = base_commands.clone()
    mix_mode = str(cfg.mix_mode)

    if mix_mode == "additive":
        residual_delta_v = float(cfg.residual_vxy_scale) * residual[..., 0:2]
        if cfg.residual_vel_clip is not None:
            residual_delta_v = torch.clamp(
                residual_delta_v,
                min=-float(cfg.residual_vel_clip),
                max=float(cfg.residual_vel_clip),
            )
        mixed[..., 0:2] = mixed[..., 0:2] + residual_delta_v
        velocity_scale = torch.ones_like(mixed[..., 0])
    elif mix_mode == "scale_velocity":
        velocity_scale = torch.clamp(
            1.0 + float(cfg.residual_vxy_scale) * residual[..., 0],
            min=float(cfg.min_velocity_scale),
            max=float(cfg.max_velocity_scale),
        )
        mixed[..., 0:2] = mixed[..., 0:2] * velocity_scale.unsqueeze(-1)
        residual_delta_v = mixed[..., 0:2] - base_commands[..., 0:2]
    else:
        raise ValueError(f"Unknown residual command mix_mode: {mix_mode}")

    heading_delta = float(cfg.residual_heading_scale) * residual[..., 2]
    if cfg.residual_heading_clip is not None:
        heading_delta = torch.clamp(
            heading_delta,
            min=-float(cfg.residual_heading_clip),
            max=float(cfg.residual_heading_clip),
        )
    mixed[..., 2] = wrap_pi(mixed[..., 2] + heading_delta)

    mixed[..., 0] = torch.clamp(
        mixed[..., 0],
        min=-float(cfg.max_forward_speed),
        max=float(cfg.max_forward_speed),
    )
    mixed[..., 1] = torch.clamp(
        mixed[..., 1],
        min=-float(cfg.max_lateral_speed),
        max=float(cfg.max_lateral_speed),
    )
    mixed[..., 2] = wrap_pi(mixed[..., 2])
    residual_delta_v = mixed[..., 0:2] - base_commands[..., 0:2]
    heading_delta = wrap_pi(mixed[..., 2] - base_commands[..., 2])

    if not return_info:
        return mixed

    base_v = base_commands[..., 0:2]
    mixed_v = mixed[..., 0:2]
    base_speed = torch.norm(base_v, dim=-1)
    mixed_speed = torch.norm(mixed_v, dim=-1)
    base_dir = base_v / torch.clamp(base_speed.unsqueeze(-1), min=1e-6)
    residual_against_base = torch.clamp(-(residual_delta_v * base_dir).sum(dim=-1), min=0.0)
    speed_ratio = mixed_speed / torch.clamp(base_speed, min=1e-6)
    info = {
        "residual_delta_v": residual_delta_v,
        "heading_delta": heading_delta,
        "velocity_scale": velocity_scale,
        "base_speed": base_speed,
        "mixed_speed": mixed_speed,
        "speed_ratio": speed_ratio,
        "anti_cancellation": residual_against_base,
    }
    return mixed, info
