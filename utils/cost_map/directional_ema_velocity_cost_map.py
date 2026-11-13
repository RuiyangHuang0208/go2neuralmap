from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from loguru import logger

from .cost_map import CostMap, CostMapCfg


@dataclass
class DirectionalEMAVelocityCostMapCfg(CostMapCfg):
    alpha: float = 0.1
    stuck_threshold_steps: int = 200
    min_velocity_norm: float = 1e-3
    min_command_norm: float = 1e-3


class DirectionalEMAVelocityCostMap(CostMap):
    """Update current-cell directional cost every step from motion heading."""

    POS_X = 0
    NEG_X = 1
    POS_Y = 2
    NEG_Y = 3

    def __init__(self, num_envs: int, cfg: DirectionalEMAVelocityCostMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {cfg.alpha}")
        if cfg.stuck_threshold_steps < 0:
            raise ValueError(f"stuck_threshold_steps must be >= 0, got {cfg.stuck_threshold_steps}")
        if cfg.min_velocity_norm < 0.0:
            raise ValueError(f"min_velocity_norm must be >= 0, got {cfg.min_velocity_norm}")
        if cfg.min_command_norm < 0.0:
            raise ValueError(f"min_command_norm must be >= 0, got {cfg.min_command_norm}")

        self.logger = logger.bind(module_name="CostMap.DirEMAVelocityCM")
        self.maps = torch.full(
            (self.num_envs, self.x_length, self.y_length, 4),
            fill_value=cfg.init_value,
            device=self.device,
            dtype=self.dtype,
        )
        self._last_indices: Tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        self._has_last_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._steps_in_current_cell = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def update(self, current_pos: torch.Tensor, new_values: torch.Tensor, **kwargs) -> torch.Tensor:
        current_pos, new_values = self._validate_update_inputs(current_pos, new_values)
        root_velocity = self._validate_planar_vector(kwargs.get("root_velocity"), "root_velocity")
        command = self._validate_planar_vector(kwargs.get("command"), "command")

        x_indices, y_indices = self._xy_to_indices(current_pos)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        last_x_indices, last_y_indices = self._last_indices
        same_cell_mask = self._has_last_indices & (x_indices == last_x_indices) & (y_indices == last_y_indices)
        moved_mask = self._has_last_indices & ~same_cell_mask

        self._steps_in_current_cell[same_cell_mask] += 1
        self._steps_in_current_cell[moved_mask] = 0
        self._steps_in_current_cell[~self._has_last_indices] = 0

        channel_weights, valid_mask = self._select_channel_weights(
            root_velocity,
            command,
            self._steps_in_current_cell,
        )
        if torch.any(valid_mask):
            alpha = float(self.cfg.alpha)
            for channel_idx in range(4):
                channel_mask = valid_mask & (channel_weights[:, channel_idx] > 0.0)
                if not torch.any(channel_mask):
                    continue

                effective_alpha = alpha * channel_weights[channel_mask, channel_idx]
                old_values = self.maps[
                    env_ids[channel_mask],
                    x_indices[channel_mask],
                    y_indices[channel_mask],
                    channel_idx,
                ]
                updated_values = effective_alpha * new_values[channel_mask] + (1.0 - effective_alpha) * old_values
                self.maps[
                    env_ids[channel_mask],
                    x_indices[channel_mask],
                    y_indices[channel_mask],
                    channel_idx,
                ] = updated_values

        self._last_indices = (x_indices, y_indices)
        self._has_last_indices[:] = True
        return valid_mask

    def get_planner_map(self, env_id: int = 0) -> torch.Tensor:
        env_id = self._validate_env_id(env_id)
        return self.maps[env_id]

    def get_traversal_cost(
        self,
        prev_grid: Tuple[int, int],
        curr_grid: Tuple[int, int],
        env_id: int = 0,
    ) -> float:
        env_id = self._validate_env_id(env_id)
        prev_x, prev_y = self._validate_grid_index(prev_grid)
        curr_x, curr_y = self._validate_grid_index(curr_grid)
        edge_id = int(self._direction_to_channel_scalar(curr_x - prev_x, curr_y - prev_y))
        return float(self.maps[env_id, prev_x, prev_y, edge_id].item())

    def reset(self, reset_buf: torch.Tensor, reset_values: bool = True) -> None:
        mask = reset_buf.to(device=self.device).view(-1) != 0
        if mask.shape[0] != self.num_envs:
            raise ValueError(f"reset_buf must have {self.num_envs} elements, got {mask.shape[0]}")
        if not torch.any(mask):
            return

        super().reset(reset_buf, reset_values=reset_values)
        self._has_last_indices[mask] = False
        self._steps_in_current_cell[mask] = 0

    def _validate_planar_vector(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} must be provided for {self.__class__.__name__}")
        value = value.to(device=self.device, dtype=self.dtype)
        if value.shape[0] != self.num_envs or value.shape[-1] < 2:
            raise ValueError(
                f"{name} must have shape ({self.num_envs}, 2+), got {tuple(value.shape)}"
            )
        return value[:, :2]

    def _select_channel_weights(
        self,
        root_velocity: torch.Tensor,
        command: torch.Tensor,
        steps_in_current_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        velocity_norm = torch.norm(root_velocity, dim=-1)
        command_norm = torch.norm(command, dim=-1)

        use_command_mask = steps_in_current_cell > int(self.cfg.stuck_threshold_steps)
        use_velocity_mask = ~use_command_mask & (velocity_norm >= float(self.cfg.min_velocity_norm))
        use_command_mask = use_command_mask & (command_norm >= float(self.cfg.min_command_norm))
        fallback_to_command_mask = (~use_velocity_mask) & (~use_command_mask) & (
            command_norm >= float(self.cfg.min_command_norm)
        )
        use_command_mask = use_command_mask | fallback_to_command_mask

        chosen_vectors = torch.zeros_like(root_velocity)
        chosen_vectors[use_velocity_mask] = root_velocity[use_velocity_mask]
        chosen_vectors[use_command_mask] = command[use_command_mask]
        valid_mask = use_velocity_mask | use_command_mask

        channel_weights = torch.zeros(self.num_envs, 4, device=self.device, dtype=self.dtype)
        if torch.any(valid_mask):
            channel_weights[valid_mask] = self._planar_vectors_to_channel_weights(chosen_vectors[valid_mask])
        return channel_weights, valid_mask

    def _planar_vectors_to_channel_weights(self, vectors: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(vectors[:, 0])
        abs_y = torch.abs(vectors[:, 1])
        total = abs_x + abs_y
        weights = torch.zeros(vectors.shape[0], 4, device=self.device, dtype=self.dtype)

        x_weights = abs_x / total
        y_weights = abs_y / total

        weights[vectors[:, 0] >= 0.0, self.POS_X] = x_weights[vectors[:, 0] >= 0.0]
        weights[vectors[:, 0] < 0.0, self.NEG_X] = x_weights[vectors[:, 0] < 0.0]
        weights[vectors[:, 1] >= 0.0, self.POS_Y] = y_weights[vectors[:, 1] >= 0.0]
        weights[vectors[:, 1] < 0.0, self.NEG_Y] = y_weights[vectors[:, 1] < 0.0]
        return weights

    def _direction_to_channel_scalar(self, delta_x: int, delta_y: int) -> int:
        if delta_x == 1 and delta_y == 0:
            return self.POS_X
        if delta_x == -1 and delta_y == 0:
            return self.NEG_X
        if delta_x == 0 and delta_y == 1:
            return self.POS_Y
        if delta_x == 0 and delta_y == -1:
            return self.NEG_Y
        raise ValueError(
            f"Traversal must be cardinal between adjacent cells, got delta=({delta_x}, {delta_y})"
        )
