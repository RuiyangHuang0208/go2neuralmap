from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from utils.diffusion_policy.dataset import (
    scale_uncertainty_crop,
)


@dataclass
class RLNavObservationConfig:
    crop_size: int = 8
    include_global_map: bool = False
    global_pool_size: int = 12
    goal_scale: float = 12.0
    velocity_scale: float = 1.5
    yaw_rate_scale: float = 1.0
    uncertainty_log_scale: float = 12.0
    uncertainty_max: float = 100000.0
    clipped_goal_distance: float = 1.0
    include_previous_command: bool = True


class RLNavObservationBuilder:
    """Build flat observations with a stable map-token layout.

    Layout:
        [map_tokens.flatten(), condition]

    Each map token contains directional latent values, scaled directional
    uncertainty, and relative geometry to the robot and goal.
    """

    STATE_FEATURE_DIM = 8

    def __init__(self, cfg: RLNavObservationConfig, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)

    @property
    def condition_dim(self) -> int:
        previous_command_dim = 3 if bool(self.cfg.include_previous_command) else 0
        return 3 + self.STATE_FEATURE_DIM + previous_command_dim

    def token_dim(self, neural_map) -> int:
        if not hasattr(neural_map, "maps"):
            latent_dim = int(neural_map.cfg.long_latent_dim) + int(neural_map.cfg.short_latent_dim)
            num_channels = 4
            directional_latent_dim = num_channels * latent_dim
        else:
            maps = neural_map.maps
            if maps.dim() == 5:
                num_channels = int(maps.shape[-2])
                latent_dim = int(maps.shape[-1])
            elif maps.dim() == 4:
                num_channels = 1
                latent_dim = int(maps.shape[-1])
            else:
                raise ValueError(f"neural_map.maps must be 4D or 5D, got shape {tuple(maps.shape)}")
            directional_latent_dim = num_channels * latent_dim
        return directional_latent_dim + num_channels + 6

    def num_tokens(self) -> int:
        crop_size = int(self.cfg.crop_size)
        return crop_size * crop_size

    def global_num_tokens(self) -> int:
        if not bool(self.cfg.include_global_map):
            return 0
        pool_size = int(self.cfg.global_pool_size)
        return pool_size * pool_size

    def observation_dim(self, neural_map) -> int:
        return (self.num_tokens() + self.global_num_tokens()) * self.token_dim(neural_map) + self.condition_dim

    def condition_observation_dim(self) -> int:
        return self.condition_dim

    @torch.no_grad()
    def build_condition_observation(
        self,
        env,
        goal_xy: torch.Tensor,
        previous_command: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.build_condition(env, goal_xy, previous_command=previous_command).to(
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def build_observation(
        self,
        env,
        neural_map,
        goal_xy: torch.Tensor,
        previous_command: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = self.build_batch(env, neural_map, goal_xy, previous_command=previous_command)
        return torch.cat(
            [
                batch["map_tokens"].reshape(env.num_envs, -1),
                batch["global_map_tokens"].reshape(env.num_envs, -1),
                batch["condition"],
            ],
            dim=-1,
        ).to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def build_batch(
        self,
        env,
        neural_map,
        goal_xy: torch.Tensor,
        previous_command: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(neural_map, "maps") or not hasattr(neural_map, "uncertainties"):
            raise TypeError("RL navigation observation requires a neural map with maps and uncertainties tensors.")

        goal_xy = goal_xy.to(device=self.device, dtype=torch.float32)
        if goal_xy.shape != (env.num_envs, 2):
            raise ValueError(f"goal_xy must have shape ({env.num_envs}, 2), got {tuple(goal_xy.shape)}")

        current_xy = env.root_local_pos[:, :2].to(device=self.device, dtype=torch.float32)
        start_grid = self.xy_to_grid(neural_map, current_xy)
        start_xy = self.grid_to_xy_batch(neural_map, start_grid, self.device)

        if hasattr(neural_map, "get_aged_uncertainties"):
            uncertainty_source = neural_map.get_aged_uncertainties()
        else:
            uncertainty_source = neural_map.uncertainties

        map_tokens, current_latents = self.build_map_tokens_batched(
            neural_map,
            uncertainty_source,
            start_grid.to(self.device),
            current_xy,
            goal_xy,
        )

        heading = self.compute_heading(env.root_states[:, 3:7].to(self.device))
        goal_direction = self.goal_direction_local(current_xy=current_xy, goal_xy=goal_xy, heading=heading)
        state_features = self.state_features(env, heading)
        condition_parts = [goal_direction, state_features]

        if bool(self.cfg.include_previous_command):
            if previous_command is None:
                previous_command = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
            condition_parts.append(self.scale_command(previous_command.to(device=self.device, dtype=torch.float32)))

        return {
            "map_tokens": map_tokens,
            "global_map_tokens": self.build_global_map_tokens_batched(
                neural_map,
                uncertainty_source,
                current_xy,
                goal_xy,
            ),
            "condition": torch.cat(condition_parts, dim=-1),
            "goal_direction_local": goal_direction,
            "state_features": state_features,
            "current_latent": current_latents,
            "start_grid": start_grid.to(self.device),
            "start_xy": start_xy,
            "heading": heading,
        }

    @torch.no_grad()
    def build_condition(
        self,
        env,
        goal_xy: torch.Tensor,
        previous_command: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        goal_xy = goal_xy.to(device=self.device, dtype=torch.float32)
        if goal_xy.shape != (env.num_envs, 2):
            raise ValueError(f"goal_xy must have shape ({env.num_envs}, 2), got {tuple(goal_xy.shape)}")

        current_xy = env.root_local_pos[:, :2].to(device=self.device, dtype=torch.float32)
        heading = self.compute_heading(env.root_states[:, 3:7].to(self.device))
        goal_direction = self.goal_direction_local(current_xy=current_xy, goal_xy=goal_xy, heading=heading)
        state_features = self.state_features(env, heading)
        condition_parts = [goal_direction, state_features]

        if bool(self.cfg.include_previous_command):
            if previous_command is None:
                previous_command = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=self.device)
            condition_parts.append(self.scale_command(previous_command.to(device=self.device, dtype=torch.float32)))

        return torch.cat(condition_parts, dim=-1)

    def build_map_tokens_batched(
        self,
        neural_map,
        uncertainty_source: torch.Tensor,
        start_grid: torch.Tensor,
        robot_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        maps = neural_map.maps.detach().to(device=self.device, dtype=torch.float32)
        uncertainty = uncertainty_source.detach().to(device=self.device, dtype=torch.float32)
        start_grid = start_grid.to(device=self.device, dtype=torch.long)
        robot_xy = robot_xy.to(device=self.device, dtype=torch.float32)
        goal_xy = goal_xy.to(device=self.device, dtype=torch.float32)

        if maps.dim() == 5:
            num_envs, _, _, num_dirs, latent_dim = maps.shape
        elif maps.dim() == 4:
            num_envs, _, _, latent_dim = maps.shape
            num_dirs = 1
        else:
            raise ValueError(f"neural_map.maps must be 4D or 5D, got shape {tuple(maps.shape)}")
        crop_size = int(self.cfg.crop_size)
        env_idx, grid_x, grid_y, valid = self.crop_indices_batched(neural_map, start_grid)

        map_crop = maps[env_idx, grid_x, grid_y]
        if map_crop.dim() == 4:
            map_crop = map_crop.unsqueeze(-2)
        uncertainty_crop = self.gather_uncertainty_crop_batched(neural_map, uncertainty, start_grid)
        if uncertainty_crop.dim() == 3:
            uncertainty_crop = uncertainty_crop.unsqueeze(-1)
        map_crop = torch.where(valid[..., None, None], map_crop, torch.zeros_like(map_crop))

        scaled_uncertainty = scale_uncertainty_crop(
            uncertainty_crop.permute(0, 3, 1, 2),
            uncertainty_log_scale=float(self.cfg.uncertainty_log_scale),
        ).permute(0, 2, 3, 1)

        latent_tokens = map_crop.reshape(num_envs, crop_size * crop_size, num_dirs * latent_dim)
        uncertainty_tokens = scaled_uncertainty.reshape(num_envs, crop_size * crop_size, num_dirs)
        cell_xy = self.crop_cell_xy_batched(neural_map, grid_x, grid_y)
        geometry = self.build_geometry_tokens(cell_xy, robot_xy, goal_xy)
        map_tokens = torch.cat([latent_tokens, uncertainty_tokens, geometry], dim=-1)

        current_latents = maps[
            torch.arange(num_envs, device=self.device, dtype=torch.long),
            start_grid[:, 0],
            start_grid[:, 1],
        ].reshape(num_envs, -1)
        return map_tokens, current_latents

    def build_global_map_tokens_batched(
        self,
        neural_map,
        uncertainty_source: torch.Tensor,
        robot_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.cfg.include_global_map):
            return torch.empty((robot_xy.shape[0], 0, self.token_dim(neural_map)), dtype=torch.float32, device=self.device)

        maps = neural_map.maps.detach().to(device=self.device, dtype=torch.float32)
        uncertainty = uncertainty_source.detach().to(device=self.device, dtype=torch.float32)
        robot_xy = robot_xy.to(device=self.device, dtype=torch.float32)
        goal_xy = goal_xy.to(device=self.device, dtype=torch.float32)

        if maps.dim() == 5:
            num_envs, x_size, y_size, num_dirs, latent_dim = maps.shape
            latent_channels = maps.permute(0, 3, 4, 1, 2).reshape(num_envs, num_dirs * latent_dim, x_size, y_size)
        elif maps.dim() == 4:
            num_envs, x_size, y_size, latent_dim = maps.shape
            num_dirs = 1
            latent_channels = maps.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"neural_map.maps must be 4D or 5D, got shape {tuple(maps.shape)}")

        if uncertainty.dim() == 3:
            uncertainty_channels = uncertainty.unsqueeze(1)
        elif uncertainty.dim() == 4:
            uncertainty_channels = uncertainty.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"neural_map.uncertainties must be 3D or 4D, got shape {tuple(uncertainty.shape)}")

        pool_size = int(self.cfg.global_pool_size)
        if pool_size <= 0:
            raise ValueError("global_pool_size must be positive")

        scaled_uncertainty = scale_uncertainty_crop(
            uncertainty_channels,
            uncertainty_log_scale=float(self.cfg.uncertainty_log_scale),
        )
        confidence = torch.clamp(1.0 - scaled_uncertainty, min=0.0, max=1.0)
        if num_dirs == 1:
            latent_confidence = confidence.expand(-1, latent_dim, -1, -1)
        else:
            latent_confidence = confidence[:, :, None].expand(-1, -1, latent_dim, -1, -1).reshape(
                num_envs,
                num_dirs * latent_dim,
                x_size,
                y_size,
            )
        pooled_confidence = F.adaptive_avg_pool2d(latent_confidence, (pool_size, pool_size))
        pooled_latent = F.adaptive_avg_pool2d(latent_channels * latent_confidence, (pool_size, pool_size)) / torch.clamp(
            pooled_confidence,
            min=1.0e-6,
        )
        pooled_uncertainty = F.adaptive_avg_pool2d(scaled_uncertainty, (pool_size, pool_size))

        latent_tokens = pooled_latent.permute(0, 2, 3, 1).reshape(num_envs, pool_size * pool_size, num_dirs * latent_dim)
        uncertainty_tokens = pooled_uncertainty.permute(0, 2, 3, 1).reshape(num_envs, pool_size * pool_size, num_dirs)
        cell_xy = self.global_pool_cell_xy(neural_map, x_size, y_size, pool_size)
        geometry = self.build_geometry_tokens(cell_xy.expand(num_envs, -1, -1), robot_xy, goal_xy)
        return torch.cat([latent_tokens, uncertainty_tokens, geometry], dim=-1)

    def crop_indices_batched(
        self,
        neural_map,
        start_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start_grid = start_grid.to(device=self.device, dtype=torch.long)
        num_envs = int(start_grid.shape[0])
        x_size = int(neural_map.maps.shape[1])
        y_size = int(neural_map.maps.shape[2])
        crop_size = int(self.cfg.crop_size)
        half = crop_size // 2
        offsets = torch.arange(crop_size, device=self.device, dtype=torch.long) - half
        raw_x = start_grid[:, 0:1] + offsets.view(1, crop_size)
        raw_y = start_grid[:, 1:2] + offsets.view(1, crop_size)
        valid_x = (raw_x >= 0) & (raw_x < x_size)
        valid_y = (raw_y >= 0) & (raw_y < y_size)
        x_idx = torch.clamp(raw_x, min=0, max=x_size - 1)
        y_idx = torch.clamp(raw_y, min=0, max=y_size - 1)
        grid_x = x_idx[:, :, None].expand(num_envs, crop_size, crop_size)
        grid_y = y_idx[:, None, :].expand(num_envs, crop_size, crop_size)
        env_idx = torch.arange(num_envs, device=self.device, dtype=torch.long)[:, None, None].expand_as(grid_x)
        valid = valid_x[:, :, None] & valid_y[:, None, :]
        return env_idx, grid_x, grid_y, valid

    def gather_uncertainty_crop_batched(
        self,
        neural_map,
        uncertainty_source: torch.Tensor,
        start_grid: torch.Tensor,
    ) -> torch.Tensor:
        uncertainty = uncertainty_source.detach().to(device=self.device, dtype=torch.float32)
        env_idx, grid_x, grid_y, valid = self.crop_indices_batched(neural_map, start_grid)
        uncertainty_crop = uncertainty[env_idx, grid_x, grid_y]
        if uncertainty_crop.dim() == 3:
            return torch.where(
                valid,
                uncertainty_crop,
                torch.full_like(uncertainty_crop, float(self.cfg.uncertainty_max)),
            )
        return torch.where(
            valid[..., None],
            uncertainty_crop,
            torch.full_like(uncertainty_crop, float(self.cfg.uncertainty_max)),
        )

    @torch.no_grad()
    def crop_uncertainty_confidence(self, env, neural_map) -> torch.Tensor:
        if not hasattr(neural_map, "uncertainties"):
            raise TypeError("RL navigation uncertainty confidence requires a neural map with uncertainties tensor.")
        current_xy = env.root_local_pos[:, :2].to(device=self.device, dtype=torch.float32)
        start_grid = self.xy_to_grid(neural_map, current_xy).to(self.device)
        if hasattr(neural_map, "get_aged_uncertainties"):
            uncertainty_source = neural_map.get_aged_uncertainties()
        else:
            uncertainty_source = neural_map.uncertainties
        uncertainty_crop = self.gather_uncertainty_crop_batched(neural_map, uncertainty_source, start_grid)
        if uncertainty_crop.dim() == 3:
            uncertainty_crop = uncertainty_crop.unsqueeze(-1)
        scaled_uncertainty = scale_uncertainty_crop(
            uncertainty_crop.permute(0, 3, 1, 2),
            uncertainty_log_scale=float(self.cfg.uncertainty_log_scale),
        )
        return 1.0 - scaled_uncertainty.mean(dim=(1, 2, 3))

    def build_map_tokens(
        self,
        map_crop: torch.Tensor,
        uncertainty_crop: torch.Tensor,
        cell_xy: torch.Tensor,
        robot_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        crop_size = int(self.cfg.crop_size)
        latent = map_crop.permute(1, 2, 0).reshape(crop_size * crop_size, -1)
        uncertainty = uncertainty_crop.permute(1, 2, 0).reshape(crop_size * crop_size, -1)
        rel_robot = (cell_xy - robot_xy.view(1, 2)) / float(self.cfg.goal_scale)
        rel_goal = (goal_xy.view(1, 2) - cell_xy) / float(self.cfg.goal_scale)
        dist_robot = torch.norm(rel_robot, dim=-1, keepdim=True)
        dist_goal = torch.clamp(
            torch.norm(goal_xy.view(1, 2) - cell_xy, dim=-1, keepdim=True) / float(self.cfg.goal_scale),
            min=0.0,
            max=float(self.cfg.clipped_goal_distance),
        )
        geometry = torch.cat([rel_robot, dist_robot, rel_goal, dist_goal], dim=-1)
        return torch.cat([latent, uncertainty, geometry], dim=-1)

    def build_geometry_tokens(
        self,
        cell_xy: torch.Tensor,
        robot_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> torch.Tensor:
        robot_xy = robot_xy[:, None, :]
        goal_xy = goal_xy[:, None, :]
        rel_robot = (cell_xy - robot_xy) / float(self.cfg.goal_scale)
        rel_goal = (goal_xy - cell_xy) / float(self.cfg.goal_scale)
        dist_robot = torch.norm(rel_robot, dim=-1, keepdim=True)
        dist_goal = torch.clamp(
            torch.norm(goal_xy - cell_xy, dim=-1, keepdim=True) / float(self.cfg.goal_scale),
            min=0.0,
            max=float(self.cfg.clipped_goal_distance),
        )
        return torch.cat([rel_robot, dist_robot, rel_goal, dist_goal], dim=-1)

    def crop_cell_xy_batched(self, neural_map, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
        x = float(neural_map.cfg.x_range[0]) + (grid_x.float() + 0.5) * float(neural_map.cfg.least_count)
        y = float(neural_map.cfg.y_range[0]) + (grid_y.float() + 0.5) * float(neural_map.cfg.least_count)
        return torch.stack([x, y], dim=-1).reshape(grid_x.shape[0], -1, 2)

    def crop_cell_xy(self, neural_map, center_grid: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        crop_size = int(self.cfg.crop_size)
        half = crop_size // 2
        offsets = torch.arange(crop_size, device=device, dtype=torch.long) - half
        x_indices = torch.clamp(center_grid[0].to(device=device) + offsets, 0, int(neural_map.maps.shape[1]) - 1)
        y_indices = torch.clamp(center_grid[1].to(device=device) + offsets, 0, int(neural_map.maps.shape[2]) - 1)
        grid_x, grid_y = torch.meshgrid(x_indices, y_indices, indexing="ij")
        x = float(neural_map.cfg.x_range[0]) + (grid_x.float() + 0.5) * float(neural_map.cfg.least_count)
        y = float(neural_map.cfg.y_range[0]) + (grid_y.float() + 0.5) * float(neural_map.cfg.least_count)
        return torch.stack([x, y], dim=-1).reshape(crop_size * crop_size, 2)

    def global_pool_cell_xy(self, neural_map, x_size: int, y_size: int, pool_size: int) -> torch.Tensor:
        x_centers = (torch.arange(pool_size, device=self.device, dtype=torch.float32) + 0.5) * (
            float(x_size) / float(pool_size)
        )
        y_centers = (torch.arange(pool_size, device=self.device, dtype=torch.float32) + 0.5) * (
            float(y_size) / float(pool_size)
        )
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
        x = float(neural_map.cfg.x_range[0]) + grid_x * float(neural_map.cfg.least_count)
        y = float(neural_map.cfg.y_range[0]) + grid_y * float(neural_map.cfg.least_count)
        return torch.stack([x, y], dim=-1).reshape(1, pool_size * pool_size, 2)

    def state_features(self, env, heading: torch.Tensor) -> torch.Tensor:
        base_lin_vel = getattr(env, "base_lin_vel", None)
        base_ang_vel = getattr(env, "base_ang_vel", None)
        if base_lin_vel is None:
            base_lin_vel = env.root_states[:, 7:10]
        if base_ang_vel is None:
            base_ang_vel = env.root_states[:, 10:13]
        base_lin_vel = base_lin_vel.to(device=self.device, dtype=torch.float32)
        base_ang_vel = base_ang_vel.to(device=self.device, dtype=torch.float32)
        return torch.cat(
            [
                base_lin_vel[:, :2] / float(self.cfg.velocity_scale),
                base_ang_vel[:, 2:3] / float(self.cfg.yaw_rate_scale),
                torch.sin(heading).unsqueeze(-1),
                torch.cos(heading).unsqueeze(-1),
                self.scale_command(env.commands[:, :3].to(device=self.device, dtype=torch.float32)),
            ],
            dim=-1,
        )

    def scale_command(self, command: torch.Tensor) -> torch.Tensor:
        scaled = command.clone()
        scaled[:, :2] = scaled[:, :2] / float(self.cfg.velocity_scale)
        scaled[:, 2] = scaled[:, 2] / float(self.cfg.yaw_rate_scale)
        return scaled

    def xy_to_grid(self, neural_map, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=neural_map.device, dtype=neural_map.dtype)
        x_idx = torch.floor((xy[:, 0] - float(neural_map.cfg.x_range[0])) / float(neural_map.cfg.least_count)).long()
        y_idx = torch.floor((xy[:, 1] - float(neural_map.cfg.y_range[0])) / float(neural_map.cfg.least_count)).long()
        x_idx = torch.clamp(x_idx, min=0, max=int(neural_map.maps.shape[1]) - 1)
        y_idx = torch.clamp(y_idx, min=0, max=int(neural_map.maps.shape[2]) - 1)
        return torch.stack([x_idx, y_idx], dim=-1)

    def grid_to_xy_batch(self, neural_map, grid_xy: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        x = float(neural_map.cfg.x_range[0]) + (grid_xy[:, 0].float() + 0.5) * float(neural_map.cfg.least_count)
        y = float(neural_map.cfg.y_range[0]) + (grid_xy[:, 1].float() + 0.5) * float(neural_map.cfg.least_count)
        return torch.stack([x, y], dim=-1).to(device=device, dtype=torch.float32)

    def goal_direction_local(
        self,
        *,
        current_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        heading: torch.Tensor,
    ) -> torch.Tensor:
        delta = goal_xy - current_xy
        distance = torch.norm(delta, dim=-1)
        unit_world = delta / torch.clamp(distance.unsqueeze(-1), min=1e-6)
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        local_x = cos_h * unit_world[:, 0] + sin_h * unit_world[:, 1]
        local_y = -sin_h * unit_world[:, 0] + cos_h * unit_world[:, 1]
        clipped_distance = torch.clamp(
            distance / float(self.cfg.goal_scale),
            min=0.0,
            max=float(self.cfg.clipped_goal_distance),
        )
        return torch.stack([local_x, local_y, clipped_distance], dim=-1)

    @staticmethod
    def compute_heading(root_quat_xyzw: torch.Tensor) -> torch.Tensor:
        qx = root_quat_xyzw[:, 0]
        qy = root_quat_xyzw[:, 1]
        qz = root_quat_xyzw[:, 2]
        qw = root_quat_xyzw[:, 3]
        forward_x = 1.0 - 2.0 * (qy.square() + qz.square())
        forward_y = 2.0 * (qx * qy + qw * qz)
        return torch.atan2(forward_y, forward_x)
