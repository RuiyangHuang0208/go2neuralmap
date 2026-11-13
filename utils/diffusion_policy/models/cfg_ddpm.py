from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ddpm import DDPMCommandDenoiser, DDPMCommandDenoiserConfig


@dataclass
class CFGDDPMCommandDenoiserConfig(DDPMCommandDenoiserConfig):
    pass


class CFGDDPMCommandDenoiser(DDPMCommandDenoiser):
    def __init__(self, cfg: CFGDDPMCommandDenoiserConfig):
        super().__init__(cfg)
        self.null_condition = nn.Parameter(torch.zeros(int(cfg.encoder.output_dim)))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        noisy_commands: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        cond_drop_mask: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        if force_uncond:
            cond = self.null_condition.view(1, -1).expand(noisy_commands.shape[0], -1)
        else:
            cond = self.encoder(batch)
        if cond_drop_mask is not None and not force_uncond:
            null_condition = self.null_condition.view(1, -1).expand(cond.shape[0], -1)
            cond = torch.where(cond_drop_mask.bool().view(-1, 1), null_condition, cond)

        return self.denoise_from_condition(cond, noisy_commands, timesteps)
