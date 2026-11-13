from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from utils.diffusion_policy.dataset import (
    crop_directional_map,
    crop_uncertainty_map,
    gather_directional_latent,
    scale_uncertainty_crop,
)


@dataclass
class ResidualObservationConfig:
    crop_size: int = 8
    goal_scale: float = 12.0
    velocity_scale: float = 1.5
    uncertainty_log_scale: float = 12.0
    uncertainty_max: float = 100000.0
    model_horizon: int = 10
    command_dim: int = 4
    clipped_goal_distance: float = 1.0


class ResidualObservationBuilder:
    def __init__(self, cfg: ResidualObservationConfig, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)

    @torch.no_grad()
    def build_batch(self, env, neural_map, goal_xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not hasattr(neural_map, "maps") or not hasattr(neural_map, "uncertainties"):
            raise TypeError(
                "Residual diffusion observation requires a directional neural map "
                "with 'maps' and 'uncertainties' tensors."
            )
        goal_xy = goal_xy.to(device=env.device, dtype=torch.float32)
        if goal_xy.shape != (env.num_envs, 2):
            raise ValueError(f"goal_xy must have shape ({env.num_envs}, 2), got {tuple(goal_xy.shape)}")

        start_grid = self.xy_to_grid(neural_map, env.root_local_pos[:, :2])
        start_xy = self.grid_to_xy_batch(neural_map, start_grid, env.device)

        map_crops = []
        uncertainty_crops = []
        current_latents = []
        if hasattr(neural_map, "get_aged_uncertainties"):
            uncertainty_source = neural_map.get_aged_uncertainties()
        else:
            uncertainty_source = neural_map.uncertainties
        for env_id in range(env.num_envs):
            map_tensor = neural_map.maps[env_id].detach().to(device=self.device, dtype=torch.float32)
            uncertainty_tensor = uncertainty_source[env_id].detach().to(device=self.device, dtype=torch.float32)
            grid = start_grid[env_id].to(self.device)
            map_crops.append(
                crop_directional_map(
                    map_tensor,
                    grid,
                    crop_size=int(self.cfg.crop_size),
                    pad_value=0.0,
                )
            )
            uncertainty_crop = crop_uncertainty_map(
                uncertainty_tensor,
                grid,
                crop_size=int(self.cfg.crop_size),
                pad_value=float(self.cfg.uncertainty_max),
            )
            uncertainty_crops.append(
                scale_uncertainty_crop(
                    uncertainty_crop,
                    uncertainty_log_scale=float(self.cfg.uncertainty_log_scale),
                )
            )
            current_latents.append(gather_directional_latent(map_tensor, grid))

        current_heading = self.compute_heading(env.root_states[:, 3:7]).to(self.device)
        current_heading_features = torch.stack(
            [torch.sin(current_heading), torch.cos(current_heading)],
            dim=-1,
        )
        goal_delta = (goal_xy.to(self.device) - start_xy.to(self.device)) / float(self.cfg.goal_scale)
        dummy_target = torch.zeros(
            (env.num_envs, int(self.cfg.model_horizon), int(self.cfg.command_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        dummy_mask = torch.ones(
            (env.num_envs, int(self.cfg.model_horizon)),
            dtype=torch.bool,
            device=self.device,
        )
        return {
            "map_crop": torch.stack(map_crops, dim=0),
            "uncertainty_crop": torch.stack(uncertainty_crops, dim=0),
            "goal_delta": goal_delta,
            "current_latent": torch.stack(current_latents, dim=0),
            "current_heading": current_heading_features,
            "target": dummy_target,
            "target_mask": dummy_mask,
            "start_grid": start_grid.to(self.device),
            "start_xy": start_xy.to(self.device),
            "heading": current_heading,
        }

    @torch.no_grad()
    def build_observation(
        self,
        env,
        neural_map,
        goal_xy: torch.Tensor,
        diffusion_encoder,
        batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = self.build_batch(env, neural_map, goal_xy)
        encoded_feature = diffusion_encoder(batch)
        goal_direction_local = self.goal_direction_local(
            current_xy=env.root_local_pos[:, :2].to(self.device),
            goal_xy=goal_xy.to(self.device),
            heading=batch["heading"],
        )
        return torch.cat(
            [
                encoded_feature,
                batch["current_latent"],
                goal_direction_local,
            ],
            dim=-1,
        )

    def observation_dim(self, neural_map, encoder_output_dim: int) -> int:
        if hasattr(neural_map, "maps"):
            latent_dim = int(neural_map.maps.shape[-2] * neural_map.maps.shape[-1])
        else:
            latent_dim = 4 * (int(neural_map.cfg.long_latent_dim) + int(neural_map.cfg.short_latent_dim))
        return int(encoder_output_dim) + latent_dim + 3

    def xy_to_grid(self, neural_map, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=neural_map.device, dtype=neural_map.dtype)
        x_idx = torch.floor((xy[:, 0] - float(neural_map.cfg.x_range[0])) / float(neural_map.cfg.least_count)).long()
        y_idx = torch.floor((xy[:, 1] - float(neural_map.cfg.y_range[0])) / float(neural_map.cfg.least_count)).long()
        x_idx = torch.clamp(x_idx, min=0, max=int(neural_map.maps.shape[1]) - 1)
        y_idx = torch.clamp(y_idx, min=0, max=int(neural_map.maps.shape[2]) - 1)
        return torch.stack([x_idx, y_idx], dim=-1)

    def grid_to_xy_batch(self, neural_map, grid_xy: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        values = [
            self.grid_to_xy(
                int(grid_xy[i, 0].item()),
                int(grid_xy[i, 1].item()),
                neural_map.cfg.x_range,
                neural_map.cfg.y_range,
                neural_map.cfg.least_count,
                device,
            )
            for i in range(grid_xy.shape[0])
        ]
        return torch.stack(values, dim=0).to(dtype=torch.float32)

    @staticmethod
    def grid_to_xy(
        x_idx: int,
        y_idx: int,
        x_range,
        y_range,
        least_count: float,
        device: torch.device | str,
    ) -> torch.Tensor:
        x = float(x_range[0]) + (float(x_idx) + 0.5) * float(least_count)
        y = float(y_range[0]) + (float(y_idx) + 0.5) * float(least_count)
        return torch.tensor([x, y], dtype=torch.float32, device=device)

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
