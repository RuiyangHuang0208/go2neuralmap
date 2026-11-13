from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from .neural_map import MapCfg, NeuralMap


@dataclass
class DirectionalBayesianNeuralMapCfg(MapCfg):
    initial_uncertainty: float = 100000.0
    incoming_uncertainty: float = 10.0
    aging_rate: float = 0.1
    max_uncertainty: float = 100000.0
    min_velocity_norm: float = 1e-3
    min_channel_weight: float = 1e-6


class DirectionalBayesianNeuralMap(NeuralMap):
    """Directional latent map with scalar Bayesian uncertainty per vector."""

    POS_X = 0
    NEG_X = 1
    POS_Y = 2
    NEG_Y = 3

    def __init__(self, num_envs: int, cfg: DirectionalBayesianNeuralMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if cfg.initial_uncertainty <= 0.0:
            raise ValueError("initial_uncertainty must be positive")
        if cfg.incoming_uncertainty <= 0.0:
            raise ValueError("incoming_uncertainty must be positive")
        if cfg.aging_rate < 0.0:
            raise ValueError("aging_rate must be non-negative")
        if cfg.max_uncertainty <= 0.0:
            raise ValueError("max_uncertainty must be positive")
        if cfg.min_velocity_norm < 0.0:
            raise ValueError("min_velocity_norm must be non-negative")
        if not 0.0 < cfg.min_channel_weight <= 1.0:
            raise ValueError("min_channel_weight must be in (0, 1]")

        self.latent_dim = cfg.long_latent_dim + cfg.short_latent_dim
        self.maps = torch.zeros(
            (self.num_envs, self.x_length, self.y_length, 4, self.latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self.uncertainties = torch.full(
            (self.num_envs, self.x_length, self.y_length, 4),
            fill_value=float(cfg.initial_uncertainty),
            device=self.device,
            dtype=self.dtype,
        )
        self.last_update_steps = torch.zeros(
            (self.num_envs, self.x_length, self.y_length, 4),
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

        root_velocity = self._validate_planar_vector(kwargs.get("root_velocity"), "root_velocity")
        channel_weights, valid_mask = self._planar_vectors_to_channel_weights(root_velocity)
        if not torch.any(valid_mask):
            self.current_step += 1
            return

        x_indices, y_indices = self._xy_to_indices(xy_coord)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        latents = torch.cat([long_latents, short_latents], dim=-1)

        for channel_idx in range(4):
            channel_mask = valid_mask & (
                channel_weights[:, channel_idx] >= float(self.cfg.min_channel_weight)
            )
            if not torch.any(channel_mask):
                continue

            weights = channel_weights[channel_mask, channel_idx]
            old_mean = self.maps[
                env_ids[channel_mask],
                x_indices[channel_mask],
                y_indices[channel_mask],
                channel_idx,
            ]
            old_uncertainty = self.uncertainties[
                env_ids[channel_mask],
                x_indices[channel_mask],
                y_indices[channel_mask],
                channel_idx,
            ]
            last_update_steps = self.last_update_steps[
                env_ids[channel_mask],
                x_indices[channel_mask],
                y_indices[channel_mask],
                channel_idx,
            ]

            step_delta = (self.current_step - last_update_steps).to(dtype=self.dtype)
            prior_uncertainty = old_uncertainty + float(self.cfg.aging_rate) * step_delta
            prior_uncertainty = torch.clamp(prior_uncertainty, max=float(self.cfg.max_uncertainty))

            observation_uncertainty = float(self.cfg.incoming_uncertainty) / weights
            gain = prior_uncertainty / (prior_uncertainty + observation_uncertainty)
            new_mean = old_mean + gain.unsqueeze(-1) * (latents[channel_mask] - old_mean)
            new_uncertainty = (1.0 - gain) * prior_uncertainty

            self.maps[
                env_ids[channel_mask],
                x_indices[channel_mask],
                y_indices[channel_mask],
                channel_idx,
            ] = new_mean
            self.uncertainties[
                env_ids[channel_mask],
                x_indices[channel_mask],
                y_indices[channel_mask],
                channel_idx,
            ] = torch.clamp(new_uncertainty, max=float(self.cfg.max_uncertainty))
            self.last_update_steps[
                env_ids[channel_mask],
                x_indices[channel_mask],
                y_indices[channel_mask],
                channel_idx,
            ] = self.current_step

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
        if direction is not None:
            return uncertainty[:, int(direction)]
        return uncertainty

    def get_aged_uncertainties(self) -> torch.Tensor:
        step_delta = (self.current_step - self.last_update_steps).to(dtype=self.dtype)
        aged = self.uncertainties + float(self.cfg.aging_rate) * step_delta
        return torch.clamp(aged, max=float(self.cfg.max_uncertainty))

    def reset(self, reset_buf: torch.Tensor, reset_cbk: None = None) -> None:
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

    def _validate_planar_vector(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} must be provided for {self.__class__.__name__}")
        value = value.to(device=self.device, dtype=self.dtype)
        if value.shape[0] != self.num_envs or value.shape[-1] < 2:
            raise ValueError(f"{name} must have shape ({self.num_envs}, 2+), got {tuple(value.shape)}")
        return value[:, :2]

    def _planar_vectors_to_channel_weights(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        velocity_norm = torch.norm(vectors, dim=-1)
        valid_mask = velocity_norm >= float(self.cfg.min_velocity_norm)
        channel_weights = torch.zeros(self.num_envs, 4, device=self.device, dtype=self.dtype)
        if not torch.any(valid_mask):
            return channel_weights, valid_mask

        valid_vectors = vectors[valid_mask]
        abs_x = torch.abs(valid_vectors[:, 0])
        abs_y = torch.abs(valid_vectors[:, 1])
        total = torch.clamp(abs_x + abs_y, min=torch.finfo(self.dtype).eps)
        x_weights = abs_x / total
        y_weights = abs_y / total

        valid_weights = torch.zeros(valid_vectors.shape[0], 4, device=self.device, dtype=self.dtype)
        valid_weights[valid_vectors[:, 0] >= 0.0, self.POS_X] = x_weights[valid_vectors[:, 0] >= 0.0]
        valid_weights[valid_vectors[:, 0] < 0.0, self.NEG_X] = x_weights[valid_vectors[:, 0] < 0.0]
        valid_weights[valid_vectors[:, 1] >= 0.0, self.POS_Y] = y_weights[valid_vectors[:, 1] >= 0.0]
        valid_weights[valid_vectors[:, 1] < 0.0, self.NEG_Y] = y_weights[valid_vectors[:, 1] < 0.0]
        channel_weights[valid_mask] = valid_weights
        return channel_weights, valid_mask
