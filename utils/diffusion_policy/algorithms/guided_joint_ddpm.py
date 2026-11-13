from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from utils.diffusion_policy.algorithms.ddpm import DDPMAlgorithmConfig
from utils.diffusion_policy.dataset import masked_mse
from utils.diffusion_policy.models import DDPMCommandDenoiser, DDPMCommandDenoiserConfig, RiskClassifier, RiskClassifierConfig
from utils.diffusion_policy.risk import LocalCropCollisionChecker, LocalCropCollisionCheckerConfig


@dataclass
class LambdaRiskScheduleConfig:
    start_step: int = 3000
    warmup_steps: int = 30000
    max_value: float = 0.1


@dataclass
class PerturbationConfig:
    velocity_noise_std: float = 0.15
    heading_noise_std: float = 0.25
    speed_scale_low: float = 0.5
    speed_scale_high: float = 1.4
    lateral_bias_std: float = 0.2


@dataclass
class CandidateSamplerConfig:
    dataset_prob: float = 0.2
    perturbation_prob: float = 0.6
    ddpm_prob: float = 0.2
    candidates_per_sample: int = 1


@dataclass
class InferenceConfig:
    sampling_steps: int = 16


@dataclass
class GuidedJointDDPMAlgorithmConfig(DDPMAlgorithmConfig):
    model: DDPMCommandDenoiserConfig
    risk_classifier: RiskClassifierConfig = field(default_factory=RiskClassifierConfig)
    collision_checker: LocalCropCollisionCheckerConfig = field(default_factory=LocalCropCollisionCheckerConfig)
    lambda_risk: LambdaRiskScheduleConfig = field(default_factory=LambdaRiskScheduleConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    candidate_sampler: CandidateSamplerConfig = field(default_factory=CandidateSamplerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ema_tau: float = 0.005
    sampling_steps: int = 16
    candidate_steps: Tuple[int, ...] = (4, 8, 12, 16)
    ddpm_candidate_count: int = 4
    perturbation_count: int = 5


class GuidedJointDDPMModel(nn.Module):
    def __init__(self, cfg: GuidedJointDDPMAlgorithmConfig):
        super().__init__()
        self.online_model = DDPMCommandDenoiser(cfg.model)
        risk_cfg = copy.deepcopy(cfg.risk_classifier)
        risk_cfg.condition_dim = int(cfg.model.encoder.output_dim)
        risk_cfg.horizon = int(cfg.model.horizon)
        risk_cfg.command_dim = int(cfg.model.command_dim)
        self.risk_classifier = RiskClassifier(risk_cfg)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_commands: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.online_model(batch, noisy_commands, timesteps)


class GuidedJointDDPMAlgorithm:
    def __init__(self, cfg: GuidedJointDDPMAlgorithmConfig):
        self.cfg = cfg
        self.model = GuidedJointDDPMModel(cfg)
        self.target_model = DDPMCommandDenoiser(cfg.model)
        self.target_model.load_state_dict(self.model.online_model.state_dict())
        self.checker = LocalCropCollisionChecker(cfg.collision_checker)
        self._buffers_initialized = False

    def to(self, device: torch.device) -> "GuidedJointDDPMAlgorithm":
        self.model.to(device)
        self.target_model.to(device)
        self.target_model.eval()
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
        self.target_model.eval()
        loss, metrics = self.loss(batch, global_step=global_step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self._ema_update_target()
        metrics["loss"] = float(loss.detach().cpu())
        return metrics

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        loss, metrics = self.loss(
            batch,
            global_step=0,
            allow_ddpm_candidates=bool(self.cfg.sample_eval_batches),
        )
        metrics["loss"] = float(loss.detach().cpu())
        return metrics

    @torch.no_grad()
    def sample(self, batch: Dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
        device = batch["target"].device
        batch_size = batch["target"].shape[0]
        shape = (batch_size, int(horizon), int(self.cfg.model.command_dim))
        x = torch.randn(shape, device=device)
        cond = self.target_model.encode_condition(batch)
        schedule = self._sampling_timestep_schedule(device)
        for iteration_idx, timestep_value in enumerate(schedule, start=1):
            timesteps = torch.full((batch_size,), int(timestep_value.item()), device=device, dtype=torch.long)
            pred_noise = self.target_model.denoise_from_condition(cond, x, timesteps)
            x0_pred = self.predict_x0(x, timesteps, pred_noise)
            x = self._ddim_previous_sample(x0_pred, pred_noise, schedule, iteration_idx, batch_size)
        return x

    @torch.no_grad()
    def infer(self, batch: Dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
        device = batch["target"].device
        batch_size = batch["target"].shape[0]
        shape = (batch_size, int(horizon), int(self.cfg.model.command_dim))
        x = torch.randn(shape, device=device)
        cond = self.model.online_model.encode_condition(batch)
        schedule = self._inference_timestep_schedule(device)
        for iteration_idx, timestep_value in enumerate(schedule, start=1):
            timesteps = torch.full((batch_size,), int(timestep_value.item()), device=device, dtype=torch.long)
            pred_noise = self.model.online_model.denoise_from_condition(cond, x, timesteps)
            x0_pred = self.predict_x0(x, timesteps, pred_noise)
            x = self._ddim_previous_sample(x0_pred, pred_noise, schedule, iteration_idx, batch_size)
        return x

    def loss(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        global_step: int,
        allow_ddpm_candidates: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        self._require_risk_batch(batch)
        target = batch["target"]
        mask = batch["target_mask"]
        batch_size = target.shape[0]

        cond_online = self.model.online_model.encode_condition(batch)
        timesteps = torch.randint(
            low=0,
            high=int(self.cfg.num_diffusion_steps),
            size=(batch_size,),
            device=target.device,
        )
        noise = torch.randn_like(target)
        noisy_target = self.q_sample(target, timesteps, noise)
        pred_noise = self.model.online_model.denoise_from_condition(cond_online, noisy_target, timesteps)
        loss_ddpm = masked_mse(pred_noise, noise, mask)

        with torch.no_grad():
            allow_ddpm = True if allow_ddpm_candidates is None else bool(allow_ddpm_candidates)
            candidates, source_ids = self._sample_candidates(batch, target, allow_ddpm=allow_ddpm)
            check = self.checker.check(
                candidates,
                batch["cost_map_crop"],
                initial_heading=batch["initial_heading"],
                target_mask=mask,
                least_count=batch["least_count"],
                velocity_command_dt=batch["velocity_command_dt"],
            )

        risk_logits = self.model.risk_classifier(cond_online, candidates)
        loss_risk = F.binary_cross_entropy_with_logits(risk_logits, check.labels)
        lambda_risk = self._lambda_risk(global_step)
        loss = loss_ddpm + float(lambda_risk) * loss_risk

        with torch.no_grad():
            risk_prob = torch.sigmoid(risk_logits)
            pred_label = risk_prob >= 0.5
            true_label = check.labels >= 0.5
            accuracy = (pred_label == true_label).float().mean()
            positive_ratio = check.labels.mean()
            source_metrics = self._source_collision_metrics(check.labels, source_ids)

        metrics = {
            "loss_ddpm": float(loss_ddpm.detach().cpu()),
            "loss_risk": float(loss_risk.detach().cpu()),
            "lambda_risk": float(lambda_risk),
            "candidate_count_per_sample": float(candidates.shape[1]),
            "candidate_dataset_fraction": float((source_ids == 0).float().mean().detach().cpu()),
            "candidate_perturbation_fraction": float((source_ids == 1).float().mean().detach().cpu()),
            "candidate_ddpm_fraction": float((source_ids == 2).float().mean().detach().cpu()),
            "risk_accuracy": float(accuracy.detach().cpu()),
            "risk_positive_ratio": float(positive_ratio.detach().cpu()),
            "checked_path_fraction": float(check.checked_path_fraction.mean().detach().cpu()),
            "out_of_crop_fraction": float(check.out_of_crop_fraction.mean().detach().cpu()),
            "max_cost_along_path": float(check.max_cost_along_path.mean().detach().cpu()),
            **source_metrics,
        }
        return loss, metrics

    @torch.no_grad()
    def sample_target_x0_candidates(
        self,
        batch: Dict[str, torch.Tensor],
        horizon: int,
        candidate_count: int,
    ) -> torch.Tensor:
        device = batch["target"].device
        batch_size = batch["target"].shape[0]
        count = int(candidate_count)
        if count <= 0:
            raise ValueError("candidate_count must be positive")
        shape = (batch_size * count, int(horizon), int(self.cfg.model.command_dim))
        repeated_batch = _repeat_batch(batch, count)
        x = torch.randn(shape, device=device)
        cond = self.target_model.encode_condition(repeated_batch)
        schedule = self._sampling_timestep_schedule(device)
        for iteration_idx, timestep_value in enumerate(schedule, start=1):
            timesteps = torch.full((shape[0],), int(timestep_value.item()), device=device, dtype=torch.long)
            pred_noise = self.target_model.denoise_from_condition(cond, x, timesteps)
            x0_pred = self.predict_x0(x, timesteps, pred_noise)
            x = self._ddim_previous_sample(x0_pred, pred_noise, schedule, iteration_idx, shape[0])
        return x.view(batch_size, count, int(horizon), int(self.cfg.model.command_dim))

    @torch.no_grad()
    def sample_target_x0_snapshots(self, batch: Dict[str, torch.Tensor], horizon: int) -> torch.Tensor:
        device = batch["target"].device
        batch_size = batch["target"].shape[0]
        shape = (batch_size, int(horizon), int(self.cfg.model.command_dim))
        x = torch.randn(shape, device=device)
        cond = self.target_model.encode_condition(batch)
        schedule = self._sampling_timestep_schedule(device)
        candidate_steps = set(int(v) for v in self.cfg.candidate_steps)
        snapshots = []
        for iteration_idx, timestep_value in enumerate(schedule, start=1):
            timesteps = torch.full((batch_size,), int(timestep_value.item()), device=device, dtype=torch.long)
            pred_noise = self.target_model.denoise_from_condition(cond, x, timesteps)
            x0_pred = self.predict_x0(x, timesteps, pred_noise)
            if iteration_idx in candidate_steps:
                snapshots.append(x0_pred)
            x = self._ddim_previous_sample(x0_pred, pred_noise, schedule, iteration_idx, batch_size)

        if len(snapshots) != int(self.cfg.ddpm_candidate_count):
            raise RuntimeError(
                f"Expected {self.cfg.ddpm_candidate_count} DDPM snapshots, got {len(snapshots)}. "
                f"candidate_steps={tuple(self.cfg.candidate_steps)} sampling_steps={self.cfg.sampling_steps}"
        )
        return torch.stack(snapshots, dim=1)

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_x0(self, x_t: torch.Tensor, timesteps: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        return (x_t - sqrt_one_minus_alpha_bar * pred_noise) / torch.clamp(sqrt_alpha_bar, min=1e-8)

    def state_dict(self) -> Dict:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        self.model.load_state_dict(state_dict["model"], strict=bool(strict))
        if "target_model" in state_dict:
            self.target_model.load_state_dict(state_dict["target_model"], strict=bool(strict))
        else:
            self.target_model.load_state_dict(self.model.online_model.state_dict(), strict=True)

    def _sampling_timestep_schedule(self, device: torch.device) -> torch.Tensor:
        steps = int(self.cfg.sampling_steps)
        if steps <= 0:
            raise ValueError("sampling_steps must be positive")
        if max(int(v) for v in self.cfg.candidate_steps) > steps:
            raise ValueError("candidate_steps cannot exceed sampling_steps")
        return self._timestep_schedule(device, steps)

    def _inference_timestep_schedule(self, device: torch.device) -> torch.Tensor:
        steps = int(self.cfg.inference.sampling_steps)
        if steps <= 0:
            raise ValueError("inference.sampling_steps must be positive")
        return self._timestep_schedule(device, steps)

    def _timestep_schedule(self, device: torch.device, steps: int) -> torch.Tensor:
        return torch.linspace(
            int(self.cfg.num_diffusion_steps) - 1,
            0,
            int(steps),
            device=device,
        ).round().long()

    def _ddim_previous_sample(
        self,
        x0_pred: torch.Tensor,
        pred_noise: torch.Tensor,
        schedule: torch.Tensor,
        iteration_idx: int,
        batch_size: int,
    ) -> torch.Tensor:
        if iteration_idx >= int(schedule.numel()):
            return x0_pred
        prev_timestep = schedule[iteration_idx]
        prev = prev_timestep.view(1).expand(batch_size)
        sqrt_alpha_bar_prev = self.sqrt_alpha_bars[prev].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_prev = self.sqrt_one_minus_alpha_bars[prev].view(-1, 1, 1)
        return sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * pred_noise

    def _sample_candidates(
        self,
        batch: Dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        allow_ddpm: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, command_dim = target.shape
        count = int(self.cfg.candidate_sampler.candidates_per_sample)
        if count <= 0:
            raise ValueError("candidate_sampler.candidates_per_sample must be positive")

        source_ids = self._sample_candidate_sources(batch_size, target.device, allow_ddpm=allow_ddpm)
        candidates = torch.empty((batch_size, count, horizon, command_dim), device=target.device, dtype=target.dtype)

        dataset_mask = source_ids == 0
        if bool(dataset_mask.any()):
            candidates[dataset_mask] = target[dataset_mask].unsqueeze(1).expand(-1, count, -1, -1)

        perturbation_mask = source_ids == 1
        if bool(perturbation_mask.any()):
            candidates[perturbation_mask] = self._make_perturbation_candidates(target[perturbation_mask], count)

        ddpm_mask = source_ids == 2
        if bool(ddpm_mask.any()):
            ddpm_batch = _index_batch(batch, ddpm_mask)
            candidates[ddpm_mask] = self.sample_target_x0_candidates(ddpm_batch, int(horizon), count)

        return candidates.detach(), source_ids

    def _sample_candidate_sources(self, batch_size: int, device: torch.device, *, allow_ddpm: bool) -> torch.Tensor:
        cfg = self.cfg.candidate_sampler
        probs = torch.tensor(
            [
                float(cfg.dataset_prob),
                float(cfg.perturbation_prob),
                float(cfg.ddpm_prob) if allow_ddpm else 0.0,
            ],
            device=device,
            dtype=torch.float32,
        )
        if bool((probs < 0.0).any()):
            raise ValueError("candidate_sampler probabilities must be non-negative")
        total = probs.sum()
        if float(total.item()) <= 0.0 and not allow_ddpm:
            probs[0] = 1.0
            total = probs.sum()
        if float(total.item()) <= 0.0:
            raise ValueError("candidate_sampler probabilities must sum to a positive value")
        probs = probs / total
        return torch.multinomial(probs, int(batch_size), replacement=True)

    def _make_perturbation_candidates(self, target: torch.Tensor, candidate_count: int) -> torch.Tensor:
        cfg = self.cfg.perturbation
        batch_size, horizon, command_dim = target.shape
        candidates = torch.empty(
            (batch_size, int(candidate_count), horizon, command_dim),
            device=target.device,
            dtype=target.dtype,
        )
        methods = torch.randint(
            low=0,
            high=int(self.cfg.perturbation_count),
            size=(batch_size, int(candidate_count)),
            device=target.device,
        )
        for sample_idx in range(batch_size):
            for candidate_idx in range(int(candidate_count)):
                method = int(methods[sample_idx, candidate_idx].item())
                candidate = target[sample_idx].clone()
                if method == 0:
                    candidate[..., 0:2] += torch.randn_like(candidate[..., 0:2]) * float(cfg.velocity_noise_std)
                elif method == 1:
                    candidate = _add_heading_noise(candidate, float(cfg.heading_noise_std))
                elif method == 2:
                    scale = torch.empty((*target.shape[1:2], 1), device=target.device).uniform_(
                        float(cfg.speed_scale_low),
                        float(cfg.speed_scale_high),
                    )
                    candidate[..., 0:2] *= scale
                elif method == 3:
                    candidate[..., 1] += torch.randn_like(candidate[..., 1]) * float(cfg.lateral_bias_std)
                else:
                    candidate[..., 0:2] = -candidate[..., 0:2]
                    candidate = _add_heading_noise(candidate, math.pi * 0.5)
                candidates[sample_idx, candidate_idx] = candidate
        return candidates

    def _lambda_risk(self, global_step: int) -> float:
        cfg = self.cfg.lambda_risk
        if int(global_step) < int(cfg.start_step):
            return 0.0
        if int(cfg.warmup_steps) <= 0:
            return float(cfg.max_value)
        progress = min((int(global_step) - int(cfg.start_step)) / float(cfg.warmup_steps), 1.0)
        return float(cfg.max_value) * 0.5 * (1.0 - math.cos(math.pi * progress))

    def _ema_update_target(self) -> None:
        tau = float(self.cfg.ema_tau)
        if not 0.0 < tau <= 1.0:
            raise ValueError("ema_tau must be in (0, 1]")
        with torch.no_grad():
            for target_param, online_param in zip(self.target_model.parameters(), self.model.online_model.parameters()):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)
            for target_buffer, online_buffer in zip(self.target_model.buffers(), self.model.online_model.buffers()):
                target_buffer.copy_(online_buffer)

    def _source_collision_metrics(self, labels: torch.Tensor, source_ids: torch.Tensor) -> Dict[str, float]:
        return {
            "risk_dataset_positive_ratio": _masked_label_mean(labels, source_ids == 0),
            "risk_perturb_positive_ratio": _masked_label_mean(labels, source_ids == 1),
            "risk_ddpm_positive_ratio": _masked_label_mean(labels, source_ids == 2),
        }

    def _require_risk_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        required = ("cost_map_crop", "initial_heading", "least_count", "velocity_command_dt")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Guided joint DDPM requires batch keys: {missing}")

    def _init_buffers(self, device: torch.device) -> None:
        betas = torch.linspace(float(self.cfg.beta_start), float(self.cfg.beta_end), int(self.cfg.num_diffusion_steps), device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self._buffers_initialized = True


def _add_heading_noise(target: torch.Tensor, std: float) -> torch.Tensor:
    out = target.clone()
    angle = torch.atan2(out[..., 2], out[..., 3])
    if abs(float(std)) > math.pi:
        angle = angle + float(std)
    else:
        angle = angle + torch.randn_like(angle) * float(std)
    out[..., 2] = torch.sin(angle)
    out[..., 3] = torch.cos(angle)
    return out


def _index_batch(batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {key: value[mask] for key, value in batch.items()}


def _repeat_batch(batch: Dict[str, torch.Tensor], repeats: int) -> Dict[str, torch.Tensor]:
    return {key: value.repeat_interleave(int(repeats), dim=0) for key, value in batch.items()}


def _masked_label_mean(labels: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(labels[mask].mean().detach().cpu())
