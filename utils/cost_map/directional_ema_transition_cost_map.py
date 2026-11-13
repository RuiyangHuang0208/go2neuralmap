from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from loguru import logger

from .cost_map import CostMap, CostMapCfg


@dataclass
class DirectionalEMATransitionCostMapCfg(CostMapCfg):
    alpha: float = 0.1


class DirectionalEMATransitionCostMap(CostMap):
    """Store outgoing 4-neighbor edge costs per cell with EMA updates on transitions."""

    POS_X = 0
    NEG_X = 1
    POS_Y = 2
    NEG_Y = 3

    def __init__(self, num_envs: int, cfg: DirectionalEMATransitionCostMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {cfg.alpha}")

        self.logger = logger.bind(module_name="CostMap.DirEMATransitionCM")
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
        self._running_sum = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)
        self._running_count = torch.zeros(self.num_envs, device=self.device, dtype=self.dtype)

    def update(self, current_pos: torch.Tensor, new_values: torch.Tensor, **kwargs) -> torch.Tensor:
        current_pos, new_values = self._validate_update_inputs(current_pos, new_values)

        x_indices, y_indices = self._xy_to_indices(current_pos)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        last_x_indices, last_y_indices = self._last_indices
        delta_x = x_indices - last_x_indices
        delta_y = y_indices - last_y_indices
        moved_mask = self._has_last_indices & ((delta_x != 0) | (delta_y != 0))

        cardinal_mask = moved_mask & (
            ((delta_x.abs() == 1) & (delta_y == 0))
            | ((delta_y.abs() == 1) & (delta_x == 0))
        )
        skipped_mask = moved_mask & ~cardinal_mask

        if torch.any(cardinal_mask):
            mean_values = self._running_sum[cardinal_mask] / torch.clamp(self._running_count[cardinal_mask], min=1.0)
            edge_ids = self._direction_to_channel(delta_x[cardinal_mask], delta_y[cardinal_mask])
            alpha = float(self.cfg.alpha)
            old_values = self.maps[
                env_ids[cardinal_mask],
                last_x_indices[cardinal_mask],
                last_y_indices[cardinal_mask],
                edge_ids,
            ]
            updated_values = alpha * mean_values + (1.0 - alpha) * old_values
            self.maps[
                env_ids[cardinal_mask],
                last_x_indices[cardinal_mask],
                last_y_indices[cardinal_mask],
                edge_ids,
            ] = updated_values

        if torch.any(skipped_mask):
            self.logger.debug(
                "Skipped non-cardinal transition updates for env ids {}",
                env_ids[skipped_mask].tolist(),
            )

        self._last_indices = (x_indices, y_indices)
        self._has_last_indices[:] = True
        self._running_sum[moved_mask] = 0.0
        self._running_count[moved_mask] = 0.0
        self._running_sum += new_values
        self._running_count += 1.0
        return cardinal_mask

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
        self._running_sum[mask] = 0.0
        self._running_count[mask] = 0.0

    def _direction_to_channel(self, delta_x: torch.Tensor, delta_y: torch.Tensor) -> torch.Tensor:
        edge_ids = torch.empty_like(delta_x, dtype=torch.long)
        edge_ids[(delta_x == 1) & (delta_y == 0)] = self.POS_X
        edge_ids[(delta_x == -1) & (delta_y == 0)] = self.NEG_X
        edge_ids[(delta_x == 0) & (delta_y == 1)] = self.POS_Y
        edge_ids[(delta_x == 0) & (delta_y == -1)] = self.NEG_Y
        return edge_ids

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
