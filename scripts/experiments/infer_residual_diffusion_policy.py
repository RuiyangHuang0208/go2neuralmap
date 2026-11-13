#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

from rsl_rl.modules import ActorCritic
from utils.diffusion_policy.dataset import training_target_to_commands
from utils.diffusion_policy.registry import build_algorithm
from utils.residual_policy.command_mixer import CommandMixerConfig, mix_residual_commands


REQUIRED_BATCH_KEYS = (
    "map_crop",
    "uncertainty_crop",
    "goal_delta",
    "current_latent",
    "current_heading",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer final velocity commands from a frozen diffusion policy plus a trained residual PPO policy."
    )
    parser.add_argument("--input", required=True, help="Path to a .pt dict containing one inference batch.")
    parser.add_argument("--output", default=None, help="Optional path to save the output .pt dict.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--diffusion-run-dir", required=True, help="Diffusion run directory containing config.yaml/latest.pt.")
    parser.add_argument("--diffusion-config", default=None, help="Override diffusion config path.")
    parser.add_argument("--diffusion-checkpoint", default=None, help="Override diffusion checkpoint path.")
    parser.add_argument("--residual-config", required=True, help="Residual training config or saved run config.yaml.")
    parser.add_argument("--residual-checkpoint", required=True, help="Residual PPO checkpoint, e.g. model_100.pt.")
    parser.add_argument("--residual-horizon", type=int, default=None, help="Override residual horizon.")
    parser.add_argument("--print-tensors", action="store_true", help="Print command tensors to stdout.")
    return parser.parse_args()


def load_diffusion(run_dir: str, config_path: str | None, checkpoint_path: str | None, device: torch.device):
    run_dir_path = Path(run_dir).expanduser()
    config = Path(config_path).expanduser() if config_path is not None else run_dir_path / "config.yaml"
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path is not None else run_dir_path / "latest.pt"
    diffusion_cfg = OmegaConf.load(config)
    algorithm = build_algorithm(diffusion_cfg.algorithm).to(device)
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    algorithm_state = loaded.get("algorithm", loaded)
    if "model" not in algorithm_state:
        raise KeyError(f"{checkpoint} does not contain algorithm.model state")
    algorithm.model.load_state_dict(algorithm_state["model"])
    algorithm.model.eval()
    for param in algorithm.model.parameters():
        param.requires_grad_(False)
    return algorithm, diffusion_cfg


def load_input_batch(path: str, diffusion_cfg, device: torch.device) -> Dict[str, torch.Tensor]:
    data = torch.load(Path(path).expanduser(), map_location=device, weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Input must be a dict saved by torch.save, got {type(data).__name__}")
    batch = {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in data.items()
    }
    if "current_heading" not in batch and "initial_heading" in batch:
        heading = batch["initial_heading"].float().view(-1)
        batch["current_heading"] = torch.stack([torch.sin(heading), torch.cos(heading)], dim=-1)
    if "heading" not in batch and "initial_heading" in batch:
        batch["heading"] = batch["initial_heading"].float().view(-1)
    if "goal_delta" not in batch and "goal_xy" in batch and "start_xy" in batch:
        goal_scale = float(diffusion_cfg.training.dataset.goal_scale)
        batch["goal_delta"] = (batch["goal_xy"].float() - batch["start_xy"].float()) / goal_scale
    if "current_xy" not in batch and "start_xy" in batch:
        batch["current_xy"] = batch["start_xy"].float()

    missing = [key for key in REQUIRED_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(f"Input is missing required diffusion batch keys: {missing}")

    batch_size = int(batch["current_latent"].shape[0])
    horizon = int(diffusion_cfg.algorithm.config.model.horizon)
    command_dim = int(diffusion_cfg.algorithm.config.model.command_dim)
    batch.setdefault("target", torch.zeros(batch_size, horizon, command_dim, device=device))
    batch.setdefault("target_mask", torch.ones(batch_size, horizon, dtype=torch.bool, device=device))
    return batch


def sample_base_commands(algorithm, batch: Dict[str, torch.Tensor], diffusion_cfg, velocity_scale: float) -> torch.Tensor:
    horizon = int(diffusion_cfg.algorithm.config.model.horizon)
    if hasattr(algorithm, "infer"):
        target = algorithm.infer(batch, horizon)
    elif hasattr(algorithm, "sample"):
        target = algorithm.sample(batch, horizon)
    else:
        target = algorithm.model(batch)
    commands = training_target_to_commands(target, velocity_scale=float(velocity_scale))
    commands[..., 2] = torch.remainder(commands[..., 2] + math.pi, 2.0 * math.pi) - math.pi
    return commands


def encode_condition(algorithm, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    model = algorithm.model
    if hasattr(model, "encode_condition"):
        return model.encode_condition(batch)
    if hasattr(model, "encoder"):
        return model.encoder(batch)
    raise TypeError("Diffusion model does not expose encode_condition(...) or encoder(...)")


def build_goal_direction_local(batch: Dict[str, torch.Tensor], goal_scale: float, clipped_goal_distance: float) -> torch.Tensor:
    if "goal_direction_local" in batch:
        return batch["goal_direction_local"].float()
    if "current_xy" not in batch or "goal_xy" not in batch:
        raise KeyError(
            "Input must include either 'goal_direction_local' or both 'current_xy' and 'goal_xy' "
            "to build residual observations."
        )
    current_xy = batch["current_xy"].float()
    goal_xy = batch["goal_xy"].float()
    if "heading" in batch:
        heading = batch["heading"].float().view(-1)
    else:
        current_heading = batch["current_heading"].float()
        heading = torch.atan2(current_heading[:, 0], current_heading[:, 1])

    delta = goal_xy - current_xy
    distance = torch.norm(delta, dim=-1)
    unit_world = delta / torch.clamp(distance.unsqueeze(-1), min=1e-6)
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    local_x = cos_h * unit_world[:, 0] + sin_h * unit_world[:, 1]
    local_y = -sin_h * unit_world[:, 0] + cos_h * unit_world[:, 1]
    clipped_distance = torch.clamp(distance / float(goal_scale), min=0.0, max=float(clipped_goal_distance))
    return torch.stack([local_x, local_y, clipped_distance], dim=-1)


def load_residual_actor(
    residual_config_path: str,
    residual_checkpoint_path: str,
    *,
    num_obs: int,
    num_actions: int,
    device: torch.device,
) -> tuple[ActorCritic, OmegaConf]:
    cfg = OmegaConf.load(Path(residual_config_path).expanduser())
    policy_cfg = OmegaConf.to_container(cfg.ppo.policy, resolve=True)
    actor_critic = ActorCritic(num_obs, num_obs, num_actions, **policy_cfg).to(device)
    checkpoint = torch.load(Path(residual_checkpoint_path).expanduser(), map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    actor_critic.load_state_dict(state_dict)
    actor_critic.eval()
    return actor_critic, cfg


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    diffusion, diffusion_cfg = load_diffusion(
        args.diffusion_run_dir,
        args.diffusion_config,
        args.diffusion_checkpoint,
        device,
    )
    batch = load_input_batch(args.input, diffusion_cfg, device)

    residual_cfg = OmegaConf.load(Path(args.residual_config).expanduser())
    dataset_cfg = diffusion_cfg.training.dataset
    residual_horizon = (
        int(args.residual_horizon)
        if args.residual_horizon is not None
        else int(residual_cfg.residual.command_mixer.residual_horizon)
    )
    command_mixer_cfg = residual_cfg.residual.command_mixer
    mixer_cfg = CommandMixerConfig(
        residual_horizon=residual_horizon,
        mix_mode=str(command_mixer_cfg.get("mix_mode", "additive")),
        residual_vxy_scale=float(command_mixer_cfg.residual_vxy_scale),
        residual_heading_scale=float(command_mixer_cfg.residual_heading_scale),
        residual_vel_clip=(
            None
            if command_mixer_cfg.get("residual_vel_clip", None) is None
            else float(command_mixer_cfg.residual_vel_clip)
        ),
        residual_heading_clip=(
            None
            if command_mixer_cfg.get("residual_heading_clip", None) is None
            else float(command_mixer_cfg.residual_heading_clip)
        ),
        min_velocity_scale=float(command_mixer_cfg.get("min_velocity_scale", 0.3)),
        max_velocity_scale=float(command_mixer_cfg.get("max_velocity_scale", 1.2)),
        max_forward_speed=float(command_mixer_cfg.max_forward_speed),
        max_lateral_speed=float(command_mixer_cfg.max_lateral_speed),
    )

    base_commands = sample_base_commands(
        diffusion,
        batch,
        diffusion_cfg,
        velocity_scale=float(dataset_cfg.velocity_scale),
    )[:, :residual_horizon]
    encoded_feature = encode_condition(diffusion, batch)
    goal_direction_local = build_goal_direction_local(
        batch,
        goal_scale=float(dataset_cfg.goal_scale),
        clipped_goal_distance=float(residual_cfg.residual.observation.clipped_goal_distance),
    ).to(device)
    residual_obs = torch.cat([encoded_feature, batch["current_latent"].float(), goal_direction_local], dim=-1)

    actor_critic, _ = load_residual_actor(
        args.residual_config,
        args.residual_checkpoint,
        num_obs=int(residual_obs.shape[-1]),
        num_actions=residual_horizon * 3,
        device=device,
    )
    residual_action = actor_critic.act_inference(residual_obs)
    final_commands = mix_residual_commands(base_commands, residual_action, mixer_cfg)

    output = {
        "base_commands": base_commands.detach().cpu(),
        "residual_action": residual_action.detach().cpu(),
        "final_commands": final_commands.detach().cpu(),
        "residual_obs": residual_obs.detach().cpu(),
        "goal_direction_local": goal_direction_local.detach().cpu(),
    }
    if args.output is not None:
        torch.save(output, Path(args.output).expanduser())
    if args.print_tensors:
        print("base_commands:", output["base_commands"])
        print("residual_action:", output["residual_action"])
        print("final_commands:", output["final_commands"])
    else:
        print(
            "Inference complete: "
            f"batch={final_commands.shape[0]}, horizon={final_commands.shape[1]}, "
            f"command_dim={final_commands.shape[2]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
