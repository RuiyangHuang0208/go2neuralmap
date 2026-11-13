from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Dict, Tuple

import torch

from utils.maps import GridMap, GridMapCfg


@dataclass
class CostMapCfg(GridMapCfg):
    init_value: float = 0.0


class CostMap(GridMap, ABC):
    """Abstract grid-aligned traversal cost memory."""

    def __init__(self, num_envs: int, cfg: CostMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        self.maps = torch.full(
            (self.num_envs, self.x_length, self.y_length, 1),
            fill_value=cfg.init_value,
            device=self.device,
            dtype=self.dtype,
        )

    @abstractmethod
    def update(self, current_pos: torch.Tensor, new_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """Update the cost map and return env-wise flags for relevant state transitions."""

    @abstractmethod
    def get_planner_map(self, env_id: int = 0) -> torch.Tensor:
        """Return the per-environment map tensor in the format expected by its planner."""

    @abstractmethod
    def get_traversal_cost(
        self,
        prev_grid: Tuple[int, int],
        curr_grid: Tuple[int, int],
        env_id: int = 0,
    ) -> float:
        """Return traversal cost for moving from prev_grid to curr_grid."""

    def reset(self, reset_buf: torch.Tensor, reset_values: bool = True) -> None:
        mask = reset_buf.to(device=self.device).view(-1) != 0
        if mask.shape[0] != self.num_envs:
            raise ValueError(f"reset_buf must have {self.num_envs} elements, got {mask.shape[0]}")
        if not torch.any(mask):
            return

        if reset_values:
            self.maps[mask] = float(self.cfg.init_value)

    def save(self, out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.maps.detach().cpu(), out_dir / "cost_map.pt")

    def load(self, in_dir: Union[str, Path]) -> None:
        in_dir = Path(in_dir)
        loaded_maps = self._load_tensor(in_dir / "cost_map.pt", self.maps.shape)
        self.maps.copy_(loaded_maps)

    def get_info(self) -> Dict:
        info = {}
        return info

    def _validate_env_id(self, env_id: int) -> int:
        env_id = int(env_id)
        if not 0 <= env_id < self.num_envs:
            raise IndexError(f"env_id must be in [0, {self.num_envs}), got {env_id}")
        return env_id

    def _validate_grid_index(self, grid: Tuple[int, int]) -> Tuple[int, int]:
        x_idx = int(grid[0])
        y_idx = int(grid[1])
        if not 0 <= x_idx < self.x_length:
            raise IndexError(f"x index out of bounds: {x_idx} not in [0, {self.x_length})")
        if not 0 <= y_idx < self.y_length:
            raise IndexError(f"y index out of bounds: {y_idx} not in [0, {self.y_length})")
        return x_idx, y_idx

    def _validate_update_inputs(self, current_pos: torch.Tensor, new_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_pos = current_pos.to(device=self.device, dtype=self.dtype)
        new_values = new_values.to(device=self.device, dtype=self.dtype).view(-1)

        if current_pos.shape != (self.num_envs, 2):
            raise ValueError(
                f"current_pos must have shape ({self.num_envs}, 2), "
                f"got {tuple(current_pos.shape)}"
            )
        if new_values.shape != (self.num_envs,):
            raise ValueError(
                f"new_values must have shape ({self.num_envs},), "
                f"got {tuple(new_values.shape)}"
            )

        return current_pos, new_values
