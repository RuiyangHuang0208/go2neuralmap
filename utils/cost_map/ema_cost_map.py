from __future__ import annotations

from dataclasses import dataclass

import torch

from .cost_map import CostMap, CostMapCfg


@dataclass
class EMACostMapCfg(CostMapCfg):
    alpha: float = 0.1


class EMACostMap(CostMap):
    """Update the current cell with an exponential moving average every step."""

    def __init__(self, num_envs: int, cfg: EMACostMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {cfg.alpha}")

    def update(self, current_pos: torch.Tensor, new_values: torch.Tensor, **kwargs) -> torch.Tensor:
        current_pos, new_values = self._validate_update_inputs(current_pos, new_values)

        x_indices, y_indices = self._xy_to_indices(current_pos)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        current_map_values = self.maps[env_ids, x_indices, y_indices, 0]
        alpha = float(self.cfg.alpha)
        updated_values = alpha * new_values + (1.0 - alpha) * current_map_values
        self.maps[env_ids, x_indices, y_indices, 0] = updated_values

        return torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

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
        super().reset(reset_buf, reset_values=reset_values)

    def get_info(self):
        info = {}
        return info
