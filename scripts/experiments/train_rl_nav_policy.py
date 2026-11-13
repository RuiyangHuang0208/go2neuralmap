#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
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

from builder import build_locomotion_model, build_neural_map, merge_config, merge_env_config  # noqa: E402
from legged_gym.utils import task_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from utils.rl_nav_policy.env import RLNavEnvConfig, RLNavPolicyEnv  # noqa: E402


def get_args():
    custom_parameters = [
        {"name": "--config_name", "type": str, "default": "rl_nav_training", "help": "Hydra config name."},
        {"name": "--my_exp_name", "type": str, "default": "go2nm", "help": "exp name."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off."},
        {"name": "--num_envs", "type": int, "help": "Number of environments."},
        {"name": "--max_iterations", "type": int, "help": "Override PPO learning iterations."},
        {"name": "--resume_checkpoint", "type": str, "help": "PPO checkpoint to load before training."},
        {"name": "--resume_optimizer", "action": "store_true", "default": False, "help": "Also load optimizer state when resuming."},
        {"name": "--logger", "type": str, "choices": ["tensorboard", "wandb", "none"], "help": "Override runner logger."},
        {"name": "--wandb_project", "type": str, "help": "Override wandb project when logger=wandb."},
        {"name": "--wandb_mode", "type": str, "help": "Override wandb mode, e.g. online/offline/disabled."},
        {"name": "--debug_observation_nan", "action": "store_true", "default": False, "help": "Check RL navigation observations after refresh and stop on NaN/Inf."},
        {"name": "--log_level", "type": str, "default": "INFO", "help": "Loguru log level."},
    ]
    args = gymutil.parse_arguments(
        description="Train a Neural Map RL navigation policy.",
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
    if args.debug_observation_nan:
        overrides.append("+rl_nav.debug_observation_nan=true")
    cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def force_yaw_rate_command_mode(env_cfg) -> None:
    if not hasattr(env_cfg, "commands"):
        return
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9


def terrain_instance_stats(env) -> dict:
    terrain_loader = getattr(env, "terrain_loader", None)
    instances = getattr(terrain_loader, "terrain_instances", None)
    if instances is None:
        instances = getattr(terrain_loader, "_instances", None)
    if not instances:
        return {}

    names = [str(getattr(instance, "name", f"terrain_{idx}")) for idx, instance in enumerate(instances)]
    counts = Counter(names)
    ordered_counts = dict(sorted(counts.items(), key=lambda item: item[0]))
    return {
        "num_instances": len(names),
        "num_unique_specs": len(ordered_counts),
        "spec_counts": ordered_counts,
        "instance_names": names,
    }


def main() -> int:
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)
    cfg_omega = load_config(args)
    logger.info("Composed RL navigation config:\n{}", OmegaConf.to_yaml(cfg_omega, resolve=False))

    env_cfg, _ = task_registry.get_cfgs(name=cfg_omega.task)
    env_cfg = merge_env_config(env_cfg, cfg_omega.env)
    force_yaw_rate_command_mode(env_cfg)
    env, _ = task_registry.make_env(name=cfg_omega.task, args=args, env_cfg=env_cfg)
    terrain_stats = terrain_instance_stats(env)
    if terrain_stats:
        OmegaConf.update(cfg_omega, "env.terrain.instantiated_specs", terrain_stats, force_add=True)
        logger.info("Instantiated terrain specs: {}", terrain_stats["spec_counts"])

    cfg_omega.modules.neural_map.num_envs = env.num_envs
    neural_map = build_neural_map(cfg_omega.modules.neural_map)
    locomotion_model = build_locomotion_model(cfg_omega.modules.locomotion_model).to(env.device)

    device = torch.device(cfg_omega.training.device)
    rl_nav_cfg = merge_config(RLNavEnvConfig(), cfg_omega.rl_nav)
    rl_nav_env = RLNavPolicyEnv(
        env=env,
        neural_map=neural_map,
        locomotion_model=locomotion_model,
        cfg=rl_nav_cfg,
        device=device,
    )

    OmegaConf.update(cfg_omega, "ppo.policy.crop_size", int(rl_nav_cfg.observation.crop_size), force_add=True)
    if str(cfg_omega.ppo.runner.policy_class_name) == "ActorCriticRLNavV3":
        OmegaConf.update(cfg_omega, "ppo.policy.local_crop_size", int(rl_nav_cfg.observation.crop_size), force_add=True)
        OmegaConf.update(cfg_omega, "ppo.policy.global_pool_size", int(rl_nav_cfg.observation.global_pool_size), force_add=True)
    OmegaConf.update(cfg_omega, "ppo.policy.token_dim", int(rl_nav_env.obs_builder.token_dim(neural_map)), force_add=True)
    OmegaConf.update(cfg_omega, "ppo.policy.condition_dim", int(rl_nav_env.obs_builder.condition_dim), force_add=True)

    log_dir = Path(cfg_omega.training.output_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg_omega, sort_keys=False))

    train_cfg = OmegaConf.to_container(cfg_omega.ppo, resolve=True)
    if terrain_stats:
        train_cfg.setdefault("runner", {})["terrain_instance_stats"] = terrain_stats
    runner = OnPolicyRunner(rl_nav_env, train_cfg, log_dir=str(log_dir), device=str(device))
    if args.resume_checkpoint is not None:
        resume_path = Path(args.resume_checkpoint).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        runner.load(str(resume_path), load_optimizer=bool(args.resume_optimizer))
        logger.info(
            "Loaded resume checkpoint={} load_optimizer={} start_iter={}",
            resume_path,
            bool(args.resume_optimizer),
            runner.current_learning_iteration,
        )
    max_iterations = int(args.max_iterations if args.max_iterations is not None else cfg_omega.training.max_iterations)
    runner.learn(max_iterations, init_at_random_ep_len=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
