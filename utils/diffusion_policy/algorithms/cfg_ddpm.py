from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from utils.diffusion_policy.algorithms.ddpm import DDPMAlgorithm, DDPMAlgorithmConfig
from utils.diffusion_policy.dataset import masked_mse
from utils.diffusion_policy.models import CFGDDPMCommandDenoiser, CFGDDPMCommandDenoiserConfig


@dataclass
class CFGDDPMAlgorithmConfig(DDPMAlgorithmConfig):
    model: CFGDDPMCommandDenoiserConfig
    condition_dropout_prob: float = 0.1
    guidance_scale: float = 2.0


class CFGDDPMAlgorithm(DDPMAlgorithm):
    def __init__(self, cfg: CFGDDPMAlgorithmConfig):
        if not 0.0 <= float(cfg.condition_dropout_prob) <= 1.0:
            raise ValueError("condition_dropout_prob must be in [0, 1]")
        if float(cfg.guidance_scale) < 0.0:
            raise ValueError("guidance_scale must be non-negative")
        self.cfg = cfg
        self.model = CFGDDPMCommandDenoiser(cfg.model)
        self._buffers_initialized = False

    def loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        target = batch["target"]
        mask = batch["target_mask"]
        batch_size = target.shape[0]
        timesteps = torch.randint(
            low=0,
            high=int(self.cfg.num_diffusion_steps),
            size=(batch_size,),
            device=target.device,
        )
        noise = torch.randn_like(target)
        noisy_target = self.q_sample(target, timesteps, noise)
        cond_drop_mask = torch.rand(batch_size, device=target.device) < float(self.cfg.condition_dropout_prob)
        pred_noise = self.model(batch, noisy_target, timesteps, cond_drop_mask=cond_drop_mask)
        loss = masked_mse(pred_noise, noise, mask)
        return loss, {
            "noise_mse": float(loss.detach().cpu()),
            "condition_drop_rate": float(cond_drop_mask.float().mean().detach().cpu()),
        }

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
        device = batch["target"].device
        shape = (batch["target"].shape[0], int(horizon), int(self.cfg.model.command_dim))
        x = torch.randn(shape, device=device)
        for step in reversed(range(int(self.cfg.num_diffusion_steps))):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred_noise = self._guided_noise_prediction(batch, x, timesteps)
            beta_t = self.betas[timesteps].view(-1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[timesteps].view(-1, 1, 1)
            mean = sqrt_recip_alpha_t * (x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t)
            if step > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[timesteps].view(-1, 1, 1)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
        return x

    def _guided_noise_prediction(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_commands: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        guidance_scale = float(self.cfg.guidance_scale)
        pred_cond = self.model(batch, noisy_commands, timesteps)
        if guidance_scale == 1.0:
            return pred_cond
        pred_uncond = self.model(batch, noisy_commands, timesteps, force_uncond=True)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
