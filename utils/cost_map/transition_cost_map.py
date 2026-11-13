from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .cost_map import CostMap, CostMapCfg
from loguru import logger

@dataclass
class TransitionCostMapCfg(CostMapCfg):
    pass


class TransitionCostMap(CostMap):
    """Accumulate per-step cost within a cell and write the mean when an env enters a new cell."""

    def __init__(self, num_envs: int, cfg: TransitionCostMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        self.logger = logger.bind(module_name="CostMap.TransitionCM")
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
        moved_mask = self._has_last_indices & ((x_indices != last_x_indices) | (y_indices != last_y_indices))

        if torch.any(moved_mask):
            mean_values = self._running_sum[moved_mask] / torch.clamp(self._running_count[moved_mask], min=1.0)
            self.logger.debug(f"Updated Cost: {mean_values}")
            self.maps[env_ids[moved_mask], last_x_indices[moved_mask], last_y_indices[moved_mask], 0] = mean_values

        self._last_indices = (x_indices, y_indices)
        self._has_last_indices[:] = True
        self._running_sum[moved_mask] = 0.0
        self._running_count[moved_mask] = 0.0
        self._running_sum += new_values
        self._running_count += 1.0
        return moved_mask

    def get_planner_map(self, env_id: int = 0) -> torch.Tensor:
        env_id = self._validate_env_id(env_id)
        return self.maps[env_id]

    def get_traversal_cost(
        self,
        prev_grid: tuple[int, int],
        curr_grid: tuple[int, int],
        env_id: int = 0,
    ) -> float:
        env_id = self._validate_env_id(env_id)
        curr_x, curr_y = self._validate_grid_index(curr_grid)
        return float(self.maps[env_id, curr_x, curr_y, 0].item())

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

    def get_info(self):
        info = {}
        return info
        pass
