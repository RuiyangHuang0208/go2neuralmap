from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch


@dataclass
class GridMapCfg:
    """Configuration shared by grid-aligned map tensors."""

    x_range: Tuple[float, float] = (0.0, 50.0)
    y_range: Tuple[float, float] = (0.0, 20.0)
    device: str = "cuda"
    dtype: str = "torch.float32"
    least_count: float = 1.0
    load_dir: Optional[str] = None


class GridMap:
    """Base utility for spatially discretized per-environment maps."""

    def __init__(self, num_envs: int, cfg: GridMapCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if cfg.dtype == "torch.float32":
            self.dtype = torch.float32
        elif cfg.dtype == "torch.float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float64
        self.num_envs = int(num_envs)

        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if cfg.least_count <= 0:
            raise ValueError("least_count must be positive")

        self.x_length, self.y_length = self._compute_grid_shape()

    def _compute_grid_shape(self) -> Tuple[int, int]:
        x_length = int(torch.ceil(torch.tensor(
            (self.cfg.x_range[1] - self.cfg.x_range[0]) / self.cfg.least_count,
            dtype=torch.float32,
        )).item())
        y_length = int(torch.ceil(torch.tensor(
            (self.cfg.y_range[1] - self.cfg.y_range[0]) / self.cfg.least_count,
            dtype=torch.float32,
        )).item())
        return x_length, y_length

    def _xy_to_indices(self, xy_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if xy_coord.shape[0] != self.num_envs or xy_coord.shape[-1] < 2:
            raise ValueError(f"xy_coord must have shape ({self.num_envs}, 2+) but got {tuple(xy_coord.shape)}")

        xy_coord = xy_coord.to(device=self.device, dtype=self.dtype)
        x_indices = torch.floor((xy_coord[:, 0] - float(self.cfg.x_range[0])) / self.cfg.least_count).to(dtype=torch.long)
        y_indices = torch.floor((xy_coord[:, 1] - float(self.cfg.y_range[0])) / self.cfg.least_count).to(dtype=torch.long)

        if (
            torch.any(x_indices < 0)
            or torch.any(x_indices >= self.x_length)
            or torch.any(y_indices < 0)
            or torch.any(y_indices >= self.y_length)
        ):
            raise IndexError(
                "xy_coord contains values outside the configured map range: "
                f"x_range={self.cfg.x_range}, y_range={self.cfg.y_range}"
            )

        return x_indices, y_indices

    def _load_tensor(
        self,
        path: Union[str, Path],
        expected_shape: torch.Size,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        path = Path(path)
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{path} must contain a torch.Tensor, got {type(tensor).__name__}")
        if tensor.shape != expected_shape:
            if (
                len(expected_shape) > 0
                and len(tensor.shape) == len(expected_shape)
                and expected_shape[0] == 1
                and tensor.shape[0] > 1
                and tensor.shape[1:] == expected_shape[1:]
            ):
                tensor = tensor[:1]
            else:
                raise ValueError(
                    f"{path} has shape {tuple(tensor.shape)}, expected {tuple(expected_shape)}"
                )
        if dtype is None:
            dtype = self.dtype
        return tensor.to(device=self.device, dtype=dtype)
