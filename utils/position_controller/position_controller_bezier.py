from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
from loguru import logger

from .position_controller import (
    PositionController,
    PositionControllerCommand,
    PositionControllerConfig,
    wrap_to_pi,
)


@dataclass
class PositionControllerBezierConfig(PositionControllerConfig):
    derivative_eval_u: float = 0.0
    tangent_scale: float = 1.0 / 6.0
    min_tangent_norm: float = 1e-4


class PositionControllerBezier(PositionController):
    def __init__(self, cfg: PositionControllerBezierConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.logger = logger.bind(module_name="PositionController.Bezier")

    def _compute_heading(self, current_xy: torch.Tensor, root_quat_xyzw: torch.Tensor) -> torch.Tensor:
        forward_vec = current_xy.new_tensor([[1.0, 0.0, 0.0]])
        forward_dir = quat_apply(root_quat_xyzw.view(1, 4), forward_vec)
        return torch.atan2(forward_dir[:, 1], forward_dir[:, 0]).squeeze(0)

    def _resolve_segment_points(
        self,
        current_xy: torch.Tensor,
        target_xy: torch.Tensor,
        waypoints: Optional[List[torch.Tensor]],
        waypoint_idx: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_xy = current_xy.view(2)
        end_xy = target_xy.view(2)

        if waypoints is None or waypoint_idx is None or len(waypoints) == 0:
            return start_xy, start_xy, end_xy, end_xy

        active_idx = max(0, min(int(waypoint_idx), len(waypoints) - 1))
        end_xy = waypoints[active_idx].view(2)

        prev_xy = start_xy
        if active_idx > 0:
            prev_xy = waypoints[active_idx - 1].view(2)

        next_xy = end_xy
        if active_idx + 1 < len(waypoints):
            next_xy = waypoints[active_idx + 1].view(2)

        return prev_xy, start_xy, end_xy, next_xy

    def _build_control_points(
        self,
        prev_xy: torch.Tensor,
        start_xy: torch.Tensor,
        end_xy: torch.Tensor,
        next_xy: torch.Tensor,
    ) -> torch.Tensor:
        tangent_scale = start_xy.new_tensor(self.cfg.tangent_scale)

        control_0 = start_xy
        control_1 = start_xy + tangent_scale * (end_xy - prev_xy)
        control_2 = end_xy - tangent_scale * (next_xy - start_xy)
        control_3 = end_xy

        return torch.stack([control_0, control_1, control_2, control_3], dim=0)

    def _eval_bezier_derivative(self, control_points: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        one_minus_u = 1.0 - u
        return (
            3.0 * one_minus_u * one_minus_u * (control_points[1] - control_points[0])
            + 6.0 * one_minus_u * u * (control_points[2] - control_points[1])
            + 3.0 * u * u * (control_points[3] - control_points[2])
        )

    def compute_command(
        self,
        current_xy: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        target_xy: torch.Tensor,
        **kwargs,
    ) -> PositionControllerCommand:
        current_xy = current_xy.view(2)
        target_xy = target_xy.view(2)
        root_quat_xyzw = root_quat_xyzw.view(1, 4)

        position_error = target_xy - current_xy
        distance = torch.norm(position_error)
        current_heading = self._compute_heading(current_xy, root_quat_xyzw)

        if distance <= self.cfg.goal_tolerance:
            zero_cmd = current_xy.new_zeros(2)
            return PositionControllerCommand(
                command_xy=zero_cmd,
                heading_target=current_heading,
                distance=distance,
                current_heading=current_heading,
                heading_error=current_heading.new_tensor(0.0),
            )

        prev_xy, start_xy, end_xy, next_xy = self._resolve_segment_points(
            current_xy=current_xy,
            target_xy=target_xy,
            waypoints=kwargs.get("waypoints"),
            waypoint_idx=kwargs.get("waypoint_idx"),
        )
        control_points = self._build_control_points(
            prev_xy=prev_xy,
            start_xy=start_xy,
            end_xy=end_xy,
            next_xy=next_xy,
        )

        eval_u = torch.clamp(current_xy.new_tensor(self.cfg.derivative_eval_u), min=0.0, max=1.0)
        velocity_world = self._eval_bezier_derivative(control_points, eval_u)

        tangent_norm = torch.norm(velocity_world)
        if tangent_norm <= self.cfg.min_tangent_norm:
            velocity_world = end_xy - current_xy
            tangent_norm = torch.norm(velocity_world)

        tangent_dir = velocity_world / torch.clamp(tangent_norm, min=self.cfg.min_tangent_norm)

        heading_target = torch.atan2(tangent_dir[1], tangent_dir[0])
        heading_error = wrap_to_pi(heading_target - current_heading)
        abs_heading_error = torch.abs(heading_error)

        if abs_heading_error >= self.cfg.turn_in_place_angle:
            zero_cmd = current_xy.new_zeros(2)
            return PositionControllerCommand(
                command_xy=zero_cmd,
                heading_target=heading_target,
                distance=distance,
                current_heading=current_heading,
                heading_error=heading_error,
            )

        heading_alignment = torch.clamp(torch.cos(heading_error), min=0.0, max=1.0)
        if abs_heading_error >= self.cfg.lateral_zero_angle:
            speed_scale = torch.clamp(heading_alignment, min=0.0, max=self.cfg.heading_slowdown)
        else:
            speed_scale = torch.clamp(heading_alignment, min=self.cfg.heading_slowdown, max=1.0)

        commanded_speed = torch.clamp(self.cfg.position_gain * distance, min=0.0, max=self.cfg.max_speed)
        velocity_world = tangent_dir * (commanded_speed * speed_scale)

        velocity_world_3d = current_xy.new_zeros((1, 3))
        velocity_world_3d[0, :2] = velocity_world
        command_xy = quat_rotate_inverse(root_quat_xyzw, velocity_world_3d)[0, :2]
        command_xy[0] = torch.clamp(command_xy[0], min=-self.cfg.max_speed, max=self.cfg.max_speed)
        command_xy[1] = torch.clamp(command_xy[1], min=-self.cfg.max_lateral_speed, max=self.cfg.max_lateral_speed)

        planar_speed = torch.norm(command_xy)
        if planar_speed > self.cfg.max_speed:
            command_xy = command_xy * (self.cfg.max_speed / planar_speed)

        return PositionControllerCommand(
            command_xy=command_xy,
            heading_target=heading_target,
            distance=distance,
            current_heading=current_heading,
            heading_error=heading_error,
        )
