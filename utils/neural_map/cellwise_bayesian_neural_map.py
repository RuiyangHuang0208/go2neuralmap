from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import torch

from .neural_map import MapCfg, NeuralMap


@dataclass
class CellwiseBayesianNeuralMapCfg(MapCfg):
    initial_uncertainty: float = 100000.0
    incoming_uncertainty: float = 10.0
    aging_rate: float = 0.1
    max_uncertainty: float = 100000.0


class CellwiseBayesianNeuralMap(NeuralMap):
    """Cell-wise latent map with scalar Bayesian uncertainty per grid cell."""

    def __init__(self, num_envs: int, cfg: CellwiseBayesianNeuralMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if cfg.initial_uncertainty <= 0.0:
            raise ValueError("initial_uncertainty must be positive")
        if cfg.incoming_uncertainty <= 0.0:
            raise ValueError("incoming_uncertainty must be positive")
        if cfg.aging_rate < 0.0:
            raise ValueError("aging_rate must be non-negative")
        if cfg.max_uncertainty <= 0.0:
            raise ValueError("max_uncertainty must be positive")

        self.latent_dim = cfg.long_latent_dim + cfg.short_latent_dim
        self.maps = torch.zeros(
            (self.num_envs, self.x_length, self.y_length, self.latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.uncertainties = torch.full(
            (self.num_envs, self.x_length, self.y_length),
            fill_value=float(cfg.initial_uncertainty),
            device=self.device,
            dtype=self.dtype,
        )
        self.last_update_steps = torch.zeros(
            (self.num_envs, self.x_length, self.y_length),
            device=self.device,
            dtype=torch.long,
        )
        self.current_step = torch.zeros((), device=self.device, dtype=torch.long)

        self.long_maps = self.maps[..., : cfg.long_latent_dim]
        self.short_maps = self.maps[..., cfg.long_latent_dim :]

    def update(
        self,
        long_latents: torch.Tensor,
        short_latents: torch.Tensor,
        xy_coord: torch.Tensor,
        **kwargs,
    ) -> None:
        long_latents = long_latents.to(device=self.device, dtype=self.dtype)
        short_latents = short_latents.to(device=self.device, dtype=self.dtype)

        if long_latents.shape != (self.num_envs, self.cfg.long_latent_dim):
            raise ValueError(
                f"long_latents must have shape ({self.num_envs}, {self.cfg.long_latent_dim}), "
                f"got {tuple(long_latents.shape)}"
            )
        if short_latents.shape != (self.num_envs, self.cfg.short_latent_dim):
            raise ValueError(
                f"short_latents must have shape ({self.num_envs}, {self.cfg.short_latent_dim}), "
                f"got {tuple(short_latents.shape)}"
            )

        x_indices, y_indices = self._xy_to_indices(xy_coord)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        latents = torch.cat([long_latents, short_latents], dim=-1)

        old_mean = self.maps[env_ids, x_indices, y_indices]
        old_uncertainty = self.uncertainties[env_ids, x_indices, y_indices]
        last_update_steps = self.last_update_steps[env_ids, x_indices, y_indices]

        step_delta = (self.current_step - last_update_steps).to(dtype=self.dtype)
        prior_uncertainty = old_uncertainty + float(self.cfg.aging_rate) * step_delta
        prior_uncertainty = torch.clamp(prior_uncertainty, max=float(self.cfg.max_uncertainty))

        observation_uncertainty = torch.full_like(prior_uncertainty, float(self.cfg.incoming_uncertainty))
        gain = prior_uncertainty / (prior_uncertainty + observation_uncertainty)
        new_mean = old_mean + gain.unsqueeze(-1) * (latents - old_mean)
        new_uncertainty = (1.0 - gain) * prior_uncertainty

        self.maps[env_ids, x_indices, y_indices] = new_mean
        self.uncertainties[env_ids, x_indices, y_indices] = torch.clamp(
            new_uncertainty,
            max=float(self.cfg.max_uncertainty),
        )
        self.last_update_steps[env_ids, x_indices, y_indices] = self.current_step

        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            if self.current_terrain_value is None:
                raise RuntimeError(
                    "current terrain labels are not registered. "
                    "Call regist_true_terrain_label(...) before update()."
                )
            self.terrain_labels[env_ids, x_indices, y_indices] = self.current_terrain_value

        self.current_step += 1

    def get_uncertainty(
        self,
        xy_coord: torch.Tensor,
        direction: Optional[int] = None,
        apply_aging: bool = True,
    ) -> torch.Tensor:
        if direction is not None:
            raise ValueError(f"{self.__class__.__name__} stores one uncertainty value per cell, not per direction")
        x_indices, y_indices = self._xy_to_indices(xy_coord)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        uncertainty = self.uncertainties[env_ids, x_indices, y_indices]
        if apply_aging:
            step_delta = (self.current_step - self.last_update_steps[env_ids, x_indices, y_indices]).to(
                dtype=self.dtype
            )
            uncertainty = torch.clamp(
                uncertainty + float(self.cfg.aging_rate) * step_delta,
                max=float(self.cfg.max_uncertainty),
            )
        return uncertainty

    def get_aged_uncertainties(self) -> torch.Tensor:
        step_delta = (self.current_step - self.last_update_steps).to(dtype=self.dtype)
        aged = self.uncertainties + float(self.cfg.aging_rate) * step_delta
        return torch.clamp(aged, max=float(self.cfg.max_uncertainty))

    def reset(self, reset_buf: torch.Tensor, reset_cbk: Optional[Callable[[], None]] = None) -> None:
        mask = reset_buf.to(device=self.device).view(-1) != 0
        if mask.shape[0] != self.num_envs:
            raise ValueError(f"reset_buf must have {self.num_envs} elements, got {mask.shape[0]}")
        if not torch.any(mask):
            return

        if reset_cbk is not None:
            reset_cbk()
        self.maps[mask] = 0.0
        self.uncertainties[mask] = float(self.cfg.initial_uncertainty)
        self.last_update_steps[mask] = self.current_step
        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            self.terrain_labels[mask] = int(self.cfg.unknown_label)  # type: ignore[arg-type]

    def save(self, out_dir: Union[str, Path]) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.maps.detach().cpu(), out_dir / "latent_map.pt")
        torch.save(self.long_maps.detach().cpu(), out_dir / "long_map.pt")
        torch.save(self.short_maps.detach().cpu(), out_dir / "short_map.pt")
        torch.save(self.uncertainties.detach().cpu(), out_dir / "uncertainty_map.pt")
        torch.save(self.last_update_steps.detach().cpu(), out_dir / "last_update_steps.pt")
        torch.save(self.current_step.detach().cpu(), out_dir / "current_step.pt")
        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            torch.save(self.terrain_labels.detach().cpu(), out_dir / "terrain_labels.pt")

    def load(self, in_dir: Union[str, Path]) -> None:
        in_dir = Path(in_dir)

        loaded_maps = self._load_tensor(in_dir / "latent_map.pt", self.maps.shape)
        loaded_uncertainties = self._load_tensor(in_dir / "uncertainty_map.pt", self.uncertainties.shape)
        loaded_last_update_steps = self._load_tensor(
            in_dir / "last_update_steps.pt",
            self.last_update_steps.shape,
            dtype=torch.long,
        )
        loaded_current_step = self._load_tensor(
            in_dir / "current_step.pt",
            self.current_step.shape,
            dtype=torch.long,
        )

        self.maps.copy_(loaded_maps)
        self.uncertainties.copy_(loaded_uncertainties)
        self.last_update_steps.copy_(loaded_last_update_steps)
        self.current_step.copy_(loaded_current_step)

        terrain_labels_path = in_dir / "terrain_labels.pt"
        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None and terrain_labels_path.exists():
            loaded_terrain_labels = self._load_tensor(
                terrain_labels_path,
                self.terrain_labels.shape,
                dtype=torch.long,
            )
            self.terrain_labels.copy_(loaded_terrain_labels)
