from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

from utils.diffusion_policy.dataset import training_target_to_commands


@dataclass
class LocalCropCollisionCheckerConfig:
    velocity_scale: float = 1.5
    obstacle_threshold: float = 70.0
    robot_radius_cells: int = 1
    interpolation_step_cells: float = 0.5


@dataclass
class CollisionCheckResult:
    labels: torch.Tensor
    checked_path_fraction: torch.Tensor
    out_of_crop_fraction: torch.Tensor
    max_cost_along_path: torch.Tensor


class LocalCropCollisionChecker:
    """Generate local-crop collision labels for candidate command sequences."""

    def __init__(self, cfg: LocalCropCollisionCheckerConfig):
        if float(cfg.velocity_scale) <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if int(cfg.robot_radius_cells) < 0:
            raise ValueError("robot_radius_cells must be non-negative")
        if float(cfg.interpolation_step_cells) <= 0.0:
            raise ValueError("interpolation_step_cells must be positive")
        self.cfg = cfg

    @torch.no_grad()
    def check(
        self,
        candidates: torch.Tensor,
        cost_map_crop: torch.Tensor,
        *,
        initial_heading: torch.Tensor,
        target_mask: torch.Tensor,
        least_count: torch.Tensor | float,
        velocity_command_dt: torch.Tensor | float,
    ) -> CollisionCheckResult:
        if candidates.dim() != 4:
            raise ValueError(f"candidates must have shape [B, K, H, 4], got {tuple(candidates.shape)}")
        if cost_map_crop.dim() != 4:
            raise ValueError(f"cost_map_crop must have shape [B, C, S, S], got {tuple(cost_map_crop.shape)}")

        batch_size, num_candidates, horizon, _ = candidates.shape
        if cost_map_crop.shape[0] != batch_size:
            raise ValueError("cost_map_crop and candidates batch size differ")
        if target_mask.shape[0] != batch_size or target_mask.shape[1] != horizon:
            raise ValueError("target_mask must have shape [B, H]")

        device = candidates.device
        obstacle_cost = cost_map_crop.float().amax(dim=1)
        commands = training_target_to_commands(candidates.float(), velocity_scale=float(self.cfg.velocity_scale))
        least_count_t = _as_batch_scalar(least_count, batch_size, device)
        dt_t = _as_batch_scalar(velocity_command_dt, batch_size, device)
        initial_heading = initial_heading.to(device=device, dtype=torch.float32).view(batch_size)
        target_mask = target_mask.to(device=device, dtype=torch.bool)

        labels = torch.zeros((batch_size, num_candidates), device=device, dtype=torch.float32)
        checked_fraction = torch.zeros_like(labels)
        out_fraction = torch.zeros_like(labels)
        max_costs = torch.zeros_like(labels)

        for batch_idx in range(batch_size):
            for candidate_idx in range(num_candidates):
                result = self._check_single(
                    commands[batch_idx, candidate_idx],
                    obstacle_cost[batch_idx],
                    initial_heading=float(initial_heading[batch_idx].item()),
                    target_mask=target_mask[batch_idx],
                    least_count=float(least_count_t[batch_idx].item()),
                    velocity_command_dt=float(dt_t[batch_idx].item()),
                )
                labels[batch_idx, candidate_idx] = result[0]
                checked_fraction[batch_idx, candidate_idx] = result[1]
                out_fraction[batch_idx, candidate_idx] = result[2]
                max_costs[batch_idx, candidate_idx] = result[3]

        return CollisionCheckResult(
            labels=labels,
            checked_path_fraction=checked_fraction,
            out_of_crop_fraction=out_fraction,
            max_cost_along_path=max_costs,
        )

    def _check_single(
        self,
        commands: torch.Tensor,
        cost_map: torch.Tensor,
        *,
        initial_heading: float,
        target_mask: torch.Tensor,
        least_count: float,
        velocity_command_dt: float,
    ) -> Tuple[float, float, float, float]:
        crop_size = int(cost_map.shape[-1])
        center = crop_size // 2
        x = float(center)
        y = float(center)
        previous_x = x
        previous_y = y
        checked_points = 0
        total_points = 0
        collided = False
        max_cost = 0.0

        valid_indices = torch.nonzero(target_mask, as_tuple=False).flatten().tolist()
        for step_idx in valid_indices:
            vx = float(commands[step_idx, 0].item())
            vy = float(commands[step_idx, 1].item())
            heading = float(commands[step_idx, 2].item())
            if not math.isfinite(heading):
                heading = initial_heading
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            dx_m = (cos_h * vx - sin_h * vy) * velocity_command_dt
            dy_m = (sin_h * vx + cos_h * vy) * velocity_command_dt
            x = previous_x + dx_m / least_count
            y = previous_y + dy_m / least_count

            distance_cells = math.hypot(x - previous_x, y - previous_y)
            sample_count = max(1, int(math.ceil(distance_cells / float(self.cfg.interpolation_step_cells))))
            for sample_idx in range(1, sample_count + 1):
                ratio = sample_idx / sample_count
                sample_x = previous_x + (x - previous_x) * ratio
                sample_y = previous_y + (y - previous_y) * ratio
                total_points += 1
                if not _in_crop(sample_x, sample_y, crop_size):
                    continue
                checked_points += 1
                sample_cost = self._footprint_max_cost(cost_map, sample_x, sample_y)
                max_cost = max(max_cost, sample_cost)
                if sample_cost >= float(self.cfg.obstacle_threshold):
                    collided = True

            previous_x = x
            previous_y = y

        if total_points == 0:
            return 0.0, 0.0, 0.0, max_cost
        checked_fraction = checked_points / total_points
        return float(collided), checked_fraction, 1.0 - checked_fraction, max_cost

    def _footprint_max_cost(self, cost_map: torch.Tensor, x: float, y: float) -> float:
        crop_size = int(cost_map.shape[-1])
        center_x = int(round(x))
        center_y = int(round(y))
        radius = int(self.cfg.robot_radius_cells)
        max_cost = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                xi = center_x + dx
                yi = center_y + dy
                if 0 <= xi < crop_size and 0 <= yi < crop_size:
                    max_cost = max(max_cost, float(cost_map[xi, yi].item()))
        return max_cost


def _in_crop(x: float, y: float, crop_size: int) -> bool:
    return 0.0 <= x < float(crop_size) and 0.0 <= y < float(crop_size)


def _as_batch_scalar(value: torch.Tensor | float, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if tensor.dim() == 0:
        return tensor.view(1).expand(batch_size)
    return tensor.view(batch_size)
