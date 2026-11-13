from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .position_controller import PositionController


GridPoint = Tuple[int, int]


@dataclass
class BezierVelocitySequenceConfig:
    horizon: int = 10
    dt: float = 0.5


def quat_xyzw_from_heading(heading: torch.Tensor) -> torch.Tensor:
    half_heading = 0.5 * heading
    quat = heading.new_zeros(4)
    quat[2] = torch.sin(half_heading)
    quat[3] = torch.cos(half_heading)
    return quat


def infer_heading(from_xy: torch.Tensor, to_xy: torch.Tensor) -> torch.Tensor:
    delta = to_xy - from_xy
    return torch.atan2(delta[1], delta[0])


def initialize_heading(
    method: str,
    current_xy: torch.Tensor,
    next_xy: torch.Tensor,
    *,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    method = method.lower()
    if method in ("path", "path_tangent", "first_path_segment"):
        return infer_heading(current_xy, next_xy)
    if method == "random":
        if rng is None:
            return current_xy.new_empty(()).uniform_(-math.pi, math.pi)
        return current_xy.new_tensor(rng.uniform(-math.pi, math.pi))
    raise ValueError(f"Unknown initial heading method: {method}")


def rotate_body_xy_to_world(command_xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    velocity_world = command_xy.new_empty(2)
    velocity_world[0] = cos_heading * command_xy[0] - sin_heading * command_xy[1]
    velocity_world[1] = sin_heading * command_xy[0] + cos_heading * command_xy[1]
    return velocity_world


def grid_to_xy(
    grid: GridPoint,
    x_range: Sequence[float],
    y_range: Sequence[float],
    least_count: float,
    device: torch.device,
) -> torch.Tensor:
    x = float(x_range[0]) + (float(grid[0]) + 0.5) * float(least_count)
    y = float(y_range[0]) + (float(grid[1]) + 0.5) * float(least_count)
    return torch.tensor([x, y], dtype=torch.float32, device=device)


def build_refined_waypoints_from_plan(
    path: Sequence[GridPoint],
    waypoint_indices: Sequence[int],
    goal_xy: torch.Tensor,
    x_range: Sequence[float],
    y_range: Sequence[float],
    least_count: float,
    device: torch.device,
) -> List[torch.Tensor]:
    waypoints = [
        grid_to_xy(path[idx], x_range, y_range, least_count, device)
        for idx in waypoint_indices
    ]
    if len(waypoints) == 0:
        return [goal_xy]

    waypoints[-1] = goal_xy
    if len(waypoints) > 1:
        waypoints = waypoints[1:]
    return waypoints


def waypoint_index_for_path_index(
    waypoint_indices: Sequence[int],
    path_index: int,
    num_waypoints: int,
) -> int:
    if num_waypoints <= 0:
        return 0

    runtime_indices = list(waypoint_indices)
    if len(runtime_indices) > 1:
        runtime_indices = runtime_indices[1:]

    if len(runtime_indices) == 0:
        return 0

    for idx, waypoint_path_idx in enumerate(runtime_indices):
        if waypoint_path_idx > path_index:
            return min(idx, num_waypoints - 1)
    return num_waypoints - 1


def compute_bezier_velocity_command_sequence(
    *,
    position_controller: PositionController,
    path: Sequence[GridPoint],
    waypoint_indices: Sequence[int],
    waypoints: Sequence[torch.Tensor],
    x_range: Sequence[float],
    y_range: Sequence[float],
    least_count: float,
    horizon: int,
    device: torch.device,
    command_dt: float = 0.5,
    initial_heading_method: str = "path",
    heading_rng: Optional[random.Random] = None,
    return_initial_heading: bool = False,
):
    """Roll out local Bezier velocity commands at a fixed time interval.

    The query states are geometric labels only. The initial state is the first
    path cell, with heading selected by ``initial_heading_method``. Each command
    is integrated in XY for ``command_dt`` seconds. No simulator rollout is used.
    """
    commands = torch.zeros((horizon, 3), dtype=torch.float32, device=device)
    mask = torch.zeros((horizon,), dtype=torch.bool, device=device)

    if len(path) < 2 or horizon <= 0 or command_dt <= 0.0:
        if return_initial_heading:
            return commands, mask, commands.new_tensor(0.0)
        return commands, mask

    current_xy = grid_to_xy(path[0], x_range, y_range, least_count, device)
    next_xy = grid_to_xy(path[1], x_range, y_range, least_count, device)
    current_heading = initialize_heading(
        initial_heading_method,
        current_xy,
        next_xy,
        rng=heading_rng,
    )
    initial_heading = current_heading.detach().clone()
    waypoint_idx = 0
    waypoint_list = list(waypoints)

    for idx in range(horizon):
        while len(waypoint_list) > 0 and waypoint_idx < len(waypoint_list) - 1:
            if torch.norm(waypoint_list[waypoint_idx] - current_xy) > position_controller.cfg.goal_tolerance:
                break
            waypoint_idx += 1

        if len(waypoint_list) > 0:
            target_xy = waypoint_list[waypoint_idx]
            reached_final_waypoint = (
                waypoint_idx == len(waypoint_list) - 1
                and torch.norm(target_xy - current_xy) <= position_controller.cfg.goal_tolerance
            )
            if reached_final_waypoint:
                break
        else:
            target_xy = next_xy

        root_quat_xyzw = quat_xyzw_from_heading(current_heading)

        output = position_controller.compute_command(
            current_xy=current_xy,
            root_quat_xyzw=root_quat_xyzw,
            target_xy=target_xy,
            waypoints=waypoint_list,
            waypoint_idx=waypoint_idx,
        )
        commands[idx, 0:2] = output.command_xy.detach()
        commands[idx, 2] = output.heading_target.detach()
        if not torch.isfinite(commands[idx, 2]):
            commands[idx, 2] = current_heading
        commands[idx, 2] = torch.remainder(commands[idx, 2] + math.pi, 2.0 * math.pi) - math.pi
        mask[idx] = True

        velocity_world = rotate_body_xy_to_world(commands[idx, 0:2], current_heading)
        current_xy = current_xy + velocity_world * float(command_dt)
        current_heading = commands[idx, 2]

    if return_initial_heading:
        return commands, mask, initial_heading
    return commands, mask
