from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from utils.maps import GridMap, GridMapCfg


@dataclass
class MapCfg(GridMapCfg):
    """Configuration for NeuralMap memory tensors."""

    long_latent_dim: int = 0
    short_latent_dim: int = 0

    collect_true_terrain_label: bool = False
    unknown_label: Optional[int] = None


class NeuralMap(GridMap, ABC):
    """Abstract neural map utility.

    Subclasses must implement ``update`` with their own map fusion strategy.
    """

    def __init__(self, num_envs: int, cfg: MapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if cfg.long_latent_dim <= 0:
            raise ValueError("long_latent_dim must be positive")
        if cfg.short_latent_dim <= 0:
            raise ValueError("short_latent_dim must be positive")

        self.long_maps = torch.zeros(
            (self.num_envs, self.x_length, self.y_length, cfg.long_latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.short_maps = torch.zeros(
            (self.num_envs, self.x_length, self.y_length, cfg.short_latent_dim),
            device=self.device,
            dtype=self.dtype,
        )

        self.current_terrain_value: Optional[torch.Tensor] = None
        self.terrain_labels: Optional[torch.Tensor] = None
        if cfg.collect_true_terrain_label:
            if not isinstance(cfg.unknown_label, int):
                raise ValueError("unknown_label must be set to int when collect_true_terrain_label=True")
            self.terrain_labels = torch.full(
                (self.num_envs, self.x_length, self.y_length),
                fill_value=cfg.unknown_label,
                device=self.device,
                dtype=torch.long,
            )

    @abstractmethod
    def update(self, long_latents: torch.Tensor, short_latents: torch.Tensor, xy_coord: torch.Tensor, **kwargs) -> None:
        """Update map values from latent features and XY coordinates."""

    def regist_true_terrain_label(self, labels: torch.Tensor) -> None:
        labels = labels.to(device=self.device, dtype=torch.long).view(-1)
        if labels.shape[0] != self.num_envs:
            raise ValueError(f"labels must have {self.num_envs} elements, got {labels.shape[0]}")
        self.current_terrain_value = labels

    def save(self, out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.long_maps.detach().cpu(), out_dir / "long_map.pt")
        torch.save(self.short_maps.detach().cpu(), out_dir / "short_map.pt")
        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            torch.save(self.terrain_labels.detach().cpu(), out_dir / "terrain_labels.pt")

    def load(self, in_dir: Union[str, Path]) -> None:
        in_dir = Path(in_dir)

        loaded_long_maps = self._load_tensor(in_dir / "long_map.pt", self.long_maps.shape)
        loaded_short_maps = self._load_tensor(in_dir / "short_map.pt", self.short_maps.shape)
        self.long_maps.copy_(loaded_long_maps)
        self.short_maps.copy_(loaded_short_maps)

        terrain_labels_path = in_dir / "terrain_labels.pt"
        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None and terrain_labels_path.exists():
            loaded_terrain_labels = self._load_tensor(
                terrain_labels_path,
                self.terrain_labels.shape,
                dtype=torch.long,
            )
            self.terrain_labels.copy_(loaded_terrain_labels)

    def reset(self, reset_buf: torch.Tensor, reset_cbk: None) -> None:
        """Reset map states for envs flagged by ``reset_buf``.

        Expected input:
        - shape ``(num_envs,)`` (or compatible flattened tensor)
        - dtype bool/int/float, where non-zero means reset
        """
        mask = reset_buf.to(device=self.device).view(-1) != 0
        if mask.shape[0] != self.num_envs:
            raise ValueError(f"reset_buf must have {self.num_envs} elements, got {mask.shape[0]}")
        if not torch.any(mask):
            return

        if reset_cbk is not None:
            reset_cbk()

        self.long_maps[mask] = 0.0
        self.short_maps[mask] = 0.0
        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            self.terrain_labels[mask] = int(self.cfg.unknown_label)  # type: ignore[arg-type]
