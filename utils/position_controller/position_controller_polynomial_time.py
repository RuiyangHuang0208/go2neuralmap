from dataclasses import dataclass
from typing import List, Optional

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
class PositionControllerPolynomialTimeConfig(PositionControllerConfig):
    fit_point_count: int = 4
    poly_degree: int = 2
    reference_speed: float = 0.6
    derivative_eval_time: float = 0.0
    min_time_horizon: float = 1e-3
    max_speed: float = 0.8
    max_lateral_speed: float = 0.5


class PositionControllerPolynomialTime(PositionController):
    def __init__(self, cfg: PositionControllerPolynomialTimeConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.logger = logger.bind(module_name="PositionController.PolynomialTime")

    def _compute_heading(self, current_xy: torch.Tensor, root_quat_xyzw: torch.Tensor) -> torch.Tensor:
        forward_vec = current_xy.new_tensor([[1.0, 0.0, 0.0]])
        forward_dir = quat_apply(root_quat_xyzw.view(1, 4), forward_vec)
        return torch.atan2(forward_dir[:, 1], forward_dir[:, 0]).squeeze(0)

    def _build_local_points(
        self,
        current_xy: torch.Tensor,
        target_xy: torch.Tensor,
        waypoints: Optional[List[torch.Tensor]],
        waypoint_idx: Optional[int],
    ) -> List[torch.Tensor]:
        points = [current_xy.view(2)]
        if waypoints is None or waypoint_idx is None:
            points.append(target_xy.view(2))
            return points

        capped_idx = max(0, waypoint_idx)
        max_idx = min(len(waypoints), capped_idx + self.cfg.fit_point_count)
        for idx in range(capped_idx, max_idx):
            points.append(waypoints[idx].view(2))

        if len(points) == 1:
            points.append(target_xy.view(2))

        return points

    def _build_time_samples(self, points: List[torch.Tensor]) -> torch.Tensor:
        times = [points[0].new_tensor(0.0)]
        cumulative_time = points[0].new_tensor(0.0)
        ref_speed = max(self.cfg.reference_speed, self.cfg.min_time_horizon)

        for idx in range(1, len(points)):
            distance = torch.norm(points[idx] - points[idx - 1])
            dt = torch.clamp(distance / ref_speed, min=self.cfg.min_time_horizon)
            cumulative_time = cumulative_time + dt
            times.append(cumulative_time)

        return torch.stack(times)

    def _fit_coefficients(self, time_samples: torch.Tensor, position_samples: torch.Tensor) -> torch.Tensor:
        degree = min(self.cfg.poly_degree, time_samples.numel() - 1)
        vandermonde = torch.vander(time_samples, N=degree + 1, increasing=False)
        return torch.linalg.lstsq(vandermonde, position_samples).solution

    def _eval_derivative(self, coeffs: torch.Tensor, eval_time: torch.Tensor) -> torch.Tensor:
        degree = coeffs.shape[0] - 1
        derivative = coeffs.new_zeros(2)
        for row_idx in range(degree):
            power = degree - row_idx
            derivative = derivative + power * coeffs[row_idx] * torch.pow(eval_time, power - 1)
        return derivative

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

        points = self._build_local_points(
            current_xy=current_xy,
            target_xy=target_xy,
            waypoints=kwargs.get("waypoints"),
            waypoint_idx=kwargs.get("waypoint_idx"),
        )
        position_samples = torch.stack(points)
        time_samples = self._build_time_samples(points)
        coeffs = self._fit_coefficients(time_samples, position_samples)

        eval_time = torch.clamp(
            current_xy.new_tensor(self.cfg.derivative_eval_time),
            min=0.0,
            max=time_samples[-1],
        )
        velocity_world = self._eval_derivative(coeffs, eval_time)
        speed = torch.norm(velocity_world)
        if speed <= self.cfg.min_time_horizon:
            velocity_world = position_error

        heading_target = torch.atan2(velocity_world[1], velocity_world[0])
        heading_error = wrap_to_pi(heading_target - current_heading)

        velocity_world_3d = current_xy.new_zeros((1, 3))
        velocity_world_3d[0, :2] = velocity_world
        command_xy = quat_rotate_inverse(root_quat_xyzw, velocity_world_3d)[0, :2]
        command_xy[0] = torch.clamp(command_xy[0], min=0.0, max=self.cfg.max_speed)
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
