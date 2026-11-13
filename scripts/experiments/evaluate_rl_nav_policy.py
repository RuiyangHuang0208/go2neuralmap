#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from isaacgym import gymutil
from isaacgym import gymapi
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
        {"name": "--run_dir", "type": str, "required": True, "help": "RL nav run directory containing config.yaml."},
        {"name": "--checkpoint", "type": str, "help": "Checkpoint path. Defaults to latest model_*.pt in run_dir."},
        {"name": "--num_envs", "type": int, "help": "Override number of envs for evaluation."},
        {"name": "--episodes", "type": int, "default": 100, "help": "Number of completed episodes to evaluate."},
        {"name": "--max_steps", "type": int, "help": "Maximum high-level eval steps. Defaults to episodes * max_episode_steps."},
        {"name": "--device", "type": str, "help": "Override torch device."},
        {"name": "--terrain_yaml", "type": str, "help": "Override terrain yaml for evaluation."},
        {"name": "--output", "type": str, "help": "Optional JSON path for metrics."},
        {"name": "--draw_goals", "action": "store_true", "default": True, "help": "Draw current goal markers in viewer."},
        {"name": "--no_draw_goals", "action": "store_false", "dest": "draw_goals", "help": "Disable goal markers."},
        {"name": "--goal_marker_radius", "type": float, "default": 0.2, "help": "Goal marker sphere radius."},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off."},
        {"name": "--render", "action": "store_false", "dest": "headless", "help": "Enable IsaacGym viewer."},
        {"name": "--log_level", "type": str, "default": "INFO", "help": "Loguru log level."},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate a trained RL navigation PPO checkpoint.",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def latest_checkpoint(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("model_*.pt"), key=model_step)
    if not paths:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")
    return paths[-1]


def model_step(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return -1


def force_yaw_rate_command_mode(env_cfg) -> None:
    if not hasattr(env_cfg, "commands"):
        return
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9


def apply_overrides(cfg_omega, args) -> None:
    if args.num_envs is not None:
        cfg_omega.env.env.num_envs = int(args.num_envs)
    if args.device is not None:
        cfg_omega.training.device = str(args.device)
        cfg_omega.env.env.device = str(args.device)
        cfg_omega.modules.neural_map.config.device = str(args.device)
    if args.terrain_yaml is not None:
        cfg_omega.env.terrain.terrains_yaml = str(args.terrain_yaml)
    cfg_omega.ppo.runner.logger = "none"
    cfg_omega.rl_nav.profiling_enabled = False


def build_eval_env(cfg_omega, args):
    env_cfg, _ = task_registry.get_cfgs(name=cfg_omega.task)
    env_cfg = merge_env_config(env_cfg, cfg_omega.env)
    force_yaw_rate_command_mode(env_cfg)
    env, _ = task_registry.make_env(name=cfg_omega.task, args=args, env_cfg=env_cfg)

    cfg_omega.modules.neural_map.num_envs = env.num_envs
    neural_map = build_neural_map(cfg_omega.modules.neural_map)
    locomotion_model = build_locomotion_model(cfg_omega.modules.locomotion_model).to(env.device)
    rl_nav_cfg = merge_config(RLNavEnvConfig(), cfg_omega.rl_nav)
    return RLNavPolicyEnv(
        env=env,
        neural_map=neural_map,
        locomotion_model=locomotion_model,
        cfg=rl_nav_cfg,
        device=torch.device(cfg_omega.training.device),
    )


def build_runner(eval_env, cfg_omega, device: torch.device) -> OnPolicyRunner:
    train_cfg = OmegaConf.to_container(cfg_omega.ppo, resolve=True)
    train_cfg["runner"]["logger"] = "none"
    return OnPolicyRunner(eval_env, train_cfg, log_dir=None, device=str(device))


@torch.no_grad()
def evaluate(
    eval_env: RLNavPolicyEnv,
    policy,
    *,
    target_episodes: int,
    max_steps: int,
    device: torch.device,
    draw_goals: bool,
    goal_marker_radius: float,
):
    obs = eval_env.get_observations().to(device)
    completed_episodes = 0
    high_level_steps = 0
    cumulative_reward = torch.zeros(eval_env.num_envs, dtype=torch.float32, device=device)
    cumulative_length = torch.zeros(eval_env.num_envs, dtype=torch.float32, device=device)
    metric_values = defaultdict(list)

    while completed_episodes < int(target_episodes) and high_level_steps < int(max_steps):
        draw_goal_markers(eval_env, enabled=draw_goals, radius=float(goal_marker_radius))
        actions = policy(obs)
        obs, _, rewards, dones, infos = eval_env.step(actions)
        obs = obs.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device, dtype=torch.bool)
        cumulative_reward += rewards
        cumulative_length += 1.0
        high_level_steps += 1

        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        episode_info = infos.get("episode", {})
        for done_id in done_ids:
            if completed_episodes >= int(target_episodes):
                break
            idx = int(done_id.item())
            metric_values["episode_reward"].append(float(cumulative_reward[idx].item()))
            metric_values["episode_length"].append(float(cumulative_length[idx].item()))
            for key, value in episode_info.items():
                if not isinstance(value, torch.Tensor) or value.numel() <= idx:
                    continue
                metric_values[key].append(float(value[idx].detach().cpu().item()))
            cumulative_reward[idx] = 0.0
            cumulative_length[idx] = 0.0
            completed_episodes += 1

    summary = {
        "completed_episodes": completed_episodes,
        "high_level_steps": high_level_steps,
        "target_episodes": int(target_episodes),
        "max_steps": int(max_steps),
    }
    for key, values in sorted(metric_values.items()):
        if not values:
            continue
        tensor = torch.tensor(values, dtype=torch.float32)
        summary[f"{key}_mean"] = float(tensor.mean().item())
        summary[f"{key}_min"] = float(tensor.min().item())
        summary[f"{key}_max"] = float(tensor.max().item())
    return summary


def draw_goal_markers(eval_env: RLNavPolicyEnv, *, enabled: bool, radius: float) -> None:
    if not enabled:
        return
    env = eval_env.env
    if not all(hasattr(env, name) for name in ("gym", "viewer", "envs")):
        return
    if env.viewer is None:
        return

    try:
        env.gym.clear_lines(env.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(
            float(radius),
            8,
            8,
            None,
            color=(1.0, 0.0, 0.0),
        )
        goal_xy = eval_env.goal_xy.to(device=env.device, dtype=torch.float32)
        for env_id in range(eval_env.num_envs):
            local_goal = torch.zeros(3, dtype=torch.float32, device=env.device)
            local_goal[:2] = goal_xy[env_id]
            terrain_id = env.env_terrain_map[env_id] if hasattr(env, "env_terrain_map") else None
            if terrain_id is not None and hasattr(env, "terrain_loader"):
                local_goal[2] = env.terrain_loader.get_height(terrain_id.view(1), local_goal[:2].view(1, 2))[0]
                world_goal = env.terrain_loader.local_to_world(terrain_id.view(1), local_goal.view(1, 3))[0]
            else:
                world_goal = local_goal
            marker_pose = gymapi.Transform(
                gymapi.Vec3(
                    float(world_goal[0].item()),
                    float(world_goal[1].item()),
                    float(world_goal[2].item()) + float(radius),
                ),
                r=None,
            )
            gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[env_id], marker_pose)
    except Exception as exc:
        logger.warning("Failed to draw goal markers: {}", exc)


def main() -> int:
    args = get_args()
    logger.remove()
    logger.add(sys.stdout, level=args.log_level)

    run_dir = Path(args.run_dir).expanduser()
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else latest_checkpoint(run_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg_omega = OmegaConf.load(config_path)
    apply_overrides(cfg_omega, args)
    OmegaConf.resolve(cfg_omega)
    device = torch.device(cfg_omega.training.device if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating run_dir={} checkpoint={}", run_dir, checkpoint_path)
    logger.info("Evaluation config:\n{}", OmegaConf.to_yaml(cfg_omega, resolve=False))

    eval_env = build_eval_env(cfg_omega, args)
    runner = build_runner(eval_env, cfg_omega, device)
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    max_steps = int(args.max_steps) if args.max_steps is not None else int(args.episodes) * int(eval_env.max_episode_length)
    summary = evaluate(
        eval_env,
        policy,
        target_episodes=int(args.episodes),
        max_steps=max_steps,
        device=device,
        draw_goals=bool(args.draw_goals),
        goal_marker_radius=float(args.goal_marker_radius),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logger.info("Saved evaluation summary to {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
