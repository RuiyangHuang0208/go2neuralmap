from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from utils.diffusion_policy.dataset import masked_heading_error, masked_mse, masked_velocity_mse
from utils.diffusion_policy.models import DDPMCommandDenoiser, DDPMCommandDenoiserConfig


@dataclass
class DDPMAlgorithmConfig:
    model: DDPMCommandDenoiserConfig
    num_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sample_eval_batches: bool = False


class DDPMAlgorithm:
    def __init__(self, cfg: DDPMAlgorithmConfig):
        self.cfg = cfg
        self.model = DDPMCommandDenoiser(cfg.model)
        self._buffers_initialized = False

    def to(self, device: torch.device) -> "DDPMAlgorithm":
        self.model.to(device)
        self._init_buffers(device)
        return self

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        global_step: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        loss, metrics = self.loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics["loss"] = float(loss.detach().cpu())
        return metrics

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        loss, metrics = self.loss(batch)
        metrics["loss"] = float(loss.detach().cpu())
        if self.cfg.sample_eval_batches:
            sample = self.sample(batch, batch["target"].shape[1])
            metrics["sample_command_mse"] = float(masked_mse(sample, batch["target"], batch["target_mask"]).cpu())
            metrics["sample_velocity_mse"] = float(masked_velocity_mse(sample, batch["target"], batch["target_mask"]).cpu())
            metrics["sample_heading_error"] = float(masked_heading_error(sample, batch["target"], batch["target_mask"]).cpu())
        return metrics

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
        pred_noise = self.model(batch, noisy_target, timesteps)
        loss = masked_mse(pred_noise, noise, mask)
        return loss, {"noise_mse": float(loss.detach().cpu())}

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
        device = batch["target"].device
        shape = (batch["target"].shape[0], int(horizon), int(self.cfg.model.command_dim))
        x = torch.randn(shape, device=device)
        for step in reversed(range(int(self.cfg.num_diffusion_steps))):
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            pred_noise = self.model(batch, x, timesteps)
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

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def state_dict(self) -> Dict:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        self.model.load_state_dict(state_dict["model"], strict=bool(strict))

    def _init_buffers(self, device: torch.device) -> None:
        betas = torch.linspace(float(self.cfg.beta_start), float(self.cfg.beta_end), int(self.cfg.num_diffusion_steps), device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous_alpha_bars = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]], dim=0)
        self.betas = betas
        self.alphas = alphas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.posterior_variance = betas * (1.0 - previous_alpha_bars) / (1.0 - alpha_bars)
        self._buffers_initialized = True
