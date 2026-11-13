from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from utils.diffusion_policy.dataset import masked_heading_error, masked_mse, masked_velocity_mse
from utils.diffusion_policy.models import BaselineCommandPolicy, BaselineCommandPolicyConfig


@dataclass
class RegressionAlgorithmConfig:
    model: BaselineCommandPolicyConfig
    sin_cos_unit_loss_weight: float = 0.01


class RegressionAlgorithm:
    def __init__(self, cfg: RegressionAlgorithmConfig):
        self.cfg = cfg
        self.model = BaselineCommandPolicy(cfg.model)

    def to(self, device: torch.device) -> "RegressionAlgorithm":
        self.model.to(device)
        return self

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        *,
        global_step: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        pred = self.model(batch)
        loss, metrics = self.loss(pred, batch["target"], batch["target_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics["loss"] = float(loss.detach().cpu())
        return metrics

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        pred = self.model(batch)
        loss, metrics = self.loss(pred, batch["target"], batch["target_mask"])
        metrics["loss"] = float(loss.detach().cpu())
        return metrics

    def loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        command_loss = masked_mse(pred, target, mask)
        unit_loss = sin_cos_unit_loss(pred, mask)
        loss = command_loss + float(self.cfg.sin_cos_unit_loss_weight) * unit_loss
        metrics = {
            "command_mse": float(command_loss.detach().cpu()),
            "velocity_mse": float(masked_velocity_mse(pred, target, mask).detach().cpu()),
            "heading_error": float(masked_heading_error(pred, target, mask).detach().cpu()),
            "sin_cos_unit_loss": float(unit_loss.detach().cpu()),
        }
        return loss, metrics

    def state_dict(self) -> Dict:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        self.model.load_state_dict(state_dict["model"], strict=bool(strict))


def sin_cos_unit_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=pred.dtype)
    radius = pred[..., 2].square() + pred[..., 3].square()
    return ((radius - 1.0).square() * valid).sum() / torch.clamp(valid.sum(), min=1.0)
