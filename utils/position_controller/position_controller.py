from dataclasses import dataclass
from typing import Optional

import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

from abc import ABC, abstractmethod


@dataclass
class PositionControllerConfig:
    goal_tolerance: float = 0.25
    position_gain: float = 0.8
    max_speed: float = 0.8
    max_lateral_speed: float = 0.5
    heading_slowdown: float = 0.35
    turn_in_place_angle: float = 2.2
    lateral_zero_angle: float = 1.2


@dataclass
class PositionControllerCommand:
    command_xy: torch.Tensor
    heading_target: torch.Tensor
    distance: torch.Tensor
    current_heading: torch.Tensor
    heading_error: torch.Tensor


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi


def sample_target_xy(
    device: torch.device,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    current_xy: Optional[torch.Tensor] = None,
    min_distance: float = 0.0,
    max_tries: int = 32,
) -> torch.Tensor:
    target_xy = None
    candidate = None

    for _ in range(max_tries):
        candidate = torch.tensor(
            [
                torch.empty(1).uniform_(x_min, x_max).item(),
                torch.empty(1).uniform_(y_min, y_max).item(),
            ],
            dtype=torch.float,
            device=device,
        )
        if current_xy is None or torch.norm(candidate - current_xy).item() >= min_distance:
            target_xy = candidate
            break

    if target_xy is None:
        target_xy = candidate

    return target_xy


def compute_turn_first_position_command(
    current_xy: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
    target_xy: torch.Tensor,
    cfg: PositionControllerConfig,
) -> PositionControllerCommand:
    current_xy = current_xy.view(2)
    target_xy = target_xy.view(2)
    root_quat_xyzw = root_quat_xyzw.view(1, 4)

    position_error = target_xy - current_xy
    distance = torch.norm(position_error)

    forward_vec = current_xy.new_tensor([[1.0, 0.0, 0.0]])
    forward_dir = quat_apply(root_quat_xyzw, forward_vec)
    current_heading = torch.atan2(forward_dir[:, 1], forward_dir[:, 0]).squeeze(0)

    if distance <= cfg.goal_tolerance:
        zero_cmd = current_xy.new_zeros(2)
        return PositionControllerCommand(
            command_xy=zero_cmd,
            heading_target=current_heading,
            distance=distance,
            current_heading=current_heading,
            heading_error=current_heading.new_tensor(0.0),
        )

    heading_target = torch.atan2(position_error[1], position_error[0])
    heading_error = wrap_to_pi(heading_target - current_heading)
    abs_heading_error = torch.abs(heading_error)

    position_error_3d = current_xy.new_zeros((1, 3))
    position_error_3d[0, :2] = position_error
    position_error_body = quat_rotate_inverse(root_quat_xyzw, position_error_3d)[0, :2]

    if abs_heading_error >= cfg.turn_in_place_angle:
        zero_cmd = current_xy.new_zeros(2)
        return PositionControllerCommand(
            command_xy=zero_cmd,
            heading_target=heading_target,
            distance=distance,
            current_heading=current_heading,
            heading_error=heading_error,
        )

    forward_alignment = torch.clamp(torch.cos(heading_error), min=0.0, max=1.0)
    if abs_heading_error >= cfg.lateral_zero_angle:
        forward_scale = torch.clamp(forward_alignment, min=0.0, max=cfg.heading_slowdown)
        lateral_scale = current_heading.new_tensor(0.0)
    else:
        forward_scale = torch.clamp(forward_alignment, min=cfg.heading_slowdown, max=1.0)
        lateral_scale = torch.clamp(1.0 - abs_heading_error / cfg.lateral_zero_angle, min=0.0, max=1.0)

    command_xy = cfg.position_gain * position_error_body
    command_xy[0] = torch.clamp(command_xy[0], min=0.0, max=cfg.max_speed) * forward_scale
    command_xy[1] = torch.clamp(command_xy[1], min=-cfg.max_lateral_speed, max=cfg.max_lateral_speed) * lateral_scale

    planar_speed = torch.norm(command_xy)
    if planar_speed > cfg.max_speed:
        command_xy = command_xy * (cfg.max_speed / planar_speed)

    return PositionControllerCommand(
        command_xy=command_xy,
        heading_target=heading_target,
        distance=distance,
        current_heading=current_heading,
        heading_error=heading_error,
    )


class PositionController(ABC):
    def __init__(self, cfg: PositionControllerConfig):
        self.cfg = cfg

    @abstractmethod
    def compute_command(
        self,
        current_xy: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        target_xy: torch.Tensor,
        **kwargs,
    ) -> PositionControllerCommand:
        pass
