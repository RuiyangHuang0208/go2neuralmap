from .neural_map import NeuralMap, MapCfg
import torch
from dataclasses import dataclass, field

@dataclass
class EMANeuralMapCfg(MapCfg):
    pass


class EMANeuralMap(NeuralMap):
    """Default neural map implementation using EMA fusion.

    For non-empty cells:
    ``new_value = blend * old + (1 - blend) * incoming``
    """

    def __init__(self, num_envs: int, cfg: MapCfg, blend: float = 0.5):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if not (0.0 <= blend <= 1.0):
            raise ValueError("blend must be in [0.0, 1.0]")
        self.blend = float(blend)

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

        current_long = self.long_maps[env_ids, x_indices, y_indices]
        current_short = self.short_maps[env_ids, x_indices, y_indices]

        is_new_long = torch.all(current_long == 0.0, dim=-1, keepdim=True)
        is_new_short = torch.all(current_short == 0.0, dim=-1, keepdim=True)

        blended_long = self.blend * current_long + (1.0 - self.blend) * long_latents
        blended_short = self.blend * current_short + (1.0 - self.blend) * short_latents

        new_long = torch.where(is_new_long, long_latents, blended_long)
        new_short = torch.where(is_new_short, short_latents, blended_short)

        self.long_maps[env_ids, x_indices, y_indices] = new_long
        self.short_maps[env_ids, x_indices, y_indices] = new_short

        if self.cfg.collect_true_terrain_label and self.terrain_labels is not None:
            if self.current_terrain_value is None:
                raise RuntimeError(
                    "current terrain labels are not registered. "
                    "Call regist_true_terrain_label(...) before update()."
                )
            self.terrain_labels[env_ids, x_indices, y_indices] = self.current_terrain_value
