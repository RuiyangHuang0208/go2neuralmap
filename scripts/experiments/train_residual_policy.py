#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
from isaacgym import gymutil
import torch
from loguru import logger
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from builder import (  # noqa: E402
    build_cost_estimator,
    build_cost_map,
    build_locomotion_model,
    build_neural_map,
    merge_config,
    merge_env_config,
)
from legged_gym.utils import task_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from utils.diffusion_policy.registry import build_algorithm  # noqa: E402
from utils.residual_policy.env import ResidualEnvConfig, ResidualPolicyEnv  # noqa: E402


def get_args():
    custom_parameters = [
        {"name": "--config_name", "type": str, "default": "residual_training", "help": "Hydra config name."},
        {"name": "--my_exp_name", "type": str, "default": "go2nm", "help": "exp name."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off."},
        {"name": "--num_envs", "type": int, "help": "Number of environments."},
        {"name": "--max_iterations", "type": int, "help": "Override PPO learning iterations."},
        {"name": "--logger", "type": str, "choices": ["tensorboard", "wandb", "none"], "help": "Override runner logger."},
        {"name": "--wandb_project", "type": str, "help": "Override wandb project when logger=wandb."},
        {"name": "--wandb_mode", "type": str, "help": "Override wandb mode, e.g. online/offline/disabled."},
        {"name": "--log_level", "type": str, "default": "INFO", "help": "Loguru log level."},
    ]
    args = gymutil.parse_arguments(
        description="Train residual policy on top of a frozen diffusion navigation policy.",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def load_config(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    hydra.initialize(config_path="configs", version_base="1.1")
    overrides = [
        f"+run_timestamp={timestamp}",
        f"+exp={args.my_exp_name}",
    ]
    if args.num_envs is not None:
        overrides.append(f"env.env.num_envs={int(args.num_envs)}")
    if args.logger is not None:
        overrides.append(f"ppo.runner.logger={args.logger}")
    if args.wandb_project is not None:
        overrides.append(f"ppo.runner.wandb_project={args.wandb_project}")
    if args.wandb_mode is not None:
        overrides.append(f"ppo.runner.wandb_mode={args.wandb_mode}")
    cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def load_diffusion_algorithm(cfg_omega, device: torch.device):
    run_dir = Path(cfg_omega.diffusion.run_dir).expanduser()
    config_path = (
        Path(cfg_omega.diffusion.config).expanduser()
        if cfg_omega.diffusion.config is not None
        else run_dir / "config.yaml"
    )
    checkpoint_path = (
        Path(cfg_omega.diffusion.checkpoint).expanduser()
        if cfg_omega.diffusion.checkpoint is not None
        else run_dir / "latest.pt"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Diffusion config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")

    diffusion_cfg = OmegaConf.load(config_path)
    algorithm = build_algorithm(diffusion_cfg.algorithm).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algorithm_state = checkpoint.get("algorithm", checkpoint)
    if "model" not in algorithm_state:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain algorithm.model state")
    algorithm.model.load_state_dict(algorithm_state["model"])
    algorithm.model.eval()
    return algorithm, diffusion_cfg, checkpoint_path


def main() -> int:
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    cfg_omega = load_config(args)
    logger.info("Composed residual config:\n{}", OmegaConf.to_yaml(cfg_omega, resolve=False))

    env_cfg, _ = task_registry.get_cfgs(name=cfg_omega.task)
    env_cfg = merge_env_config(env_cfg, cfg_omega.env)
    env, _ = task_registry.make_env(name=cfg_omega.task, args=args, env_cfg=env_cfg)

    cfg_omega.modules.neural_map.num_envs = env.num_envs
    cfg_omega.modules.cost_map.num_envs = env.num_envs
    neural_map = build_neural_map(cfg_omega.modules.neural_map)
    cost_map = build_cost_map(cfg_omega.modules.cost_map)
    locomotion_model = build_locomotion_model(cfg_omega.modules.locomotion_model).to(env.device)
    cost_estimator = build_cost_estimator(cfg_omega.modules.cost_estimator, env)

    device = torch.device(cfg_omega.training.device)
    diffusion_algorithm, diffusion_cfg, diffusion_checkpoint = load_diffusion_algorithm(cfg_omega, device)
    logger.info("Loaded frozen diffusion checkpoint: {}", diffusion_checkpoint)

    residual_cfg = merge_config(ResidualEnvConfig(), cfg_omega.residual)
    dataset_cfg = diffusion_cfg.training.dataset
    residual_cfg.observation.crop_size = int(dataset_cfg.crop_size)
    residual_cfg.observation.goal_scale = float(dataset_cfg.goal_scale)
    residual_cfg.observation.velocity_scale = float(dataset_cfg.velocity_scale)
    residual_cfg.observation.uncertainty_log_scale = float(dataset_cfg.uncertainty_log_scale)
    residual_cfg.observation.uncertainty_max = float(dataset_cfg.uncertainty_max)
    residual_cfg.observation.model_horizon = int(diffusion_cfg.algorithm.config.model.horizon)
    residual_cfg.observation.command_dim = int(diffusion_cfg.algorithm.config.model.command_dim)
    if residual_cfg.command_mixer.residual_horizon > residual_cfg.observation.model_horizon:
        raise ValueError("residual_horizon cannot exceed diffusion model horizon")

    residual_env = ResidualPolicyEnv(
        env=env,
        neural_map=neural_map,
        cost_map=cost_map,
        locomotion_model=locomotion_model,
        cost_estimator=cost_estimator,
        diffusion_algorithm=diffusion_algorithm,
        cfg=residual_cfg,
        device=device,
    )

    log_dir = Path(cfg_omega.training.output_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg_omega, sort_keys=False))

    train_cfg = OmegaConf.to_container(cfg_omega.ppo, resolve=True)
    runner = OnPolicyRunner(residual_env, train_cfg, log_dir=str(log_dir), device=str(device))
    max_iterations = int(args.max_iterations if args.max_iterations is not None else cfg_omega.training.max_iterations)
    runner.learn(max_iterations, init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
