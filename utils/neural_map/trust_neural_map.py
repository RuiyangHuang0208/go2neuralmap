from .neural_map import MapCfg, NeuralMap
import torch
from dataclasses import dataclass, field


@dataclass
class TrustNeuralMapCfg(MapCfg):
    pass


class TrustNeuralMap(NeuralMap):
    """Neural map implementation that fully trusts current collected latents.

    This subclass directly overwrites map cells with incoming latents without any blending.
    """

    def update(self, long_latents: torch.Tensor, short_latents: torch.Tensor, xy_coord: torch.Tensor, **kwargs) -> None:
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

        self.long_maps[env_ids, x_indices, y_indices] = long_latents
        self.short_maps[env_ids, x_indices, y_indices] = short_latents

        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            if self.current_terrain_value is None:
                raise RuntimeError(
                    "current terrain labels are not registered. "
                    "Call regist_true_terrain_label(...) before update()."
                )
            self.terrain_labels[env_ids, x_indices, y_indices] = self.current_terrain_value
