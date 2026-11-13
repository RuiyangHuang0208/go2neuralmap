from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from isaacgym import gymapi, gymutil
import torch
from loguru import logger
from omegaconf import OmegaConf

from rsl_rl.modules import ActorCritic
from utils.diffusion_policy.dataset import training_target_to_commands
from utils.diffusion_policy.registry import build_algorithm
from utils.residual_policy.command_mixer import CommandMixerConfig, mix_residual_commands, wrap_pi
from utils.residual_policy.observation import ResidualObservationBuilder, ResidualObservationConfig

from .vel_command_generator import VelCommandGenerator, VelCommandGeneratorConfig


@dataclass
class ResidualDiffusionPolicyVelCommandGeneratorConfig(VelCommandGeneratorConfig):
    diffusion_run_dir: str = "logs/diffusion_policy/cfg_ddpm_2026-05-11_22-27-03"
    diffusion_checkpoint: Optional[str] = None
    diffusion_config: Optional[str] = None
    residual_run_dir: str = "logs/residual_policy/2026-05-12_09-56-44"
    residual_checkpoint: Optional[str] = None
    residual_config: Optional[str] = None
    device: str = "cuda"
    execution_horizon: int = 2
    command_dt: float = 0.5
    main_loop_hz: float = 50.0
    goal_tolerance: float = 0.35
    resample_goal_flag: bool = True
    reset_env_flag: bool = False
    goal_source: str = "sampled"
    mix_mode: str = "additive"
    residual_vxy_scale: float = 0.3
    residual_heading_scale: float = 0.4
    residual_vel_clip: Optional[float] = None
    residual_heading_clip: Optional[float] = None
    min_velocity_scale: float = 0.3
    max_velocity_scale: float = 1.2
    max_forward_speed: float = 1.5
    max_lateral_speed: float = 0.5


@dataclass
class ResidualDiffusionPolicyRunningInfo:
    current_xy: Optional[torch.Tensor] = None
    current_grid_xy: Optional[tuple[int, int]] = None
    goal_xy: Optional[torch.Tensor] = None
    sequence_index: int = 0
    command_step_index: int = 0
    sampled_sequence: bool = False
    reached_goal_flag: bool = False
    traj_cost: float = 0.0
    reached_goal_count: int = 0
    mean_distance_to_goal: float = 0.0
    mean_residual_norm: float = 0.0
    execution_horizon: int = 0
    steps_per_command: int = 0


class ResidualDiffusionPolicyVelCommandGenerator(VelCommandGenerator):
    def __init__(self, cfg: ResidualDiffusionPolicyVelCommandGeneratorConfig, context=None, **kwargs):
        self.logger = logger.bind(module_name="VelCmdGen.ResidualDiffusionPolicy")
        if context is None:
            raise ValueError("No context is passed")
        if cfg.goal_source not in ("sampled", "user"):
            raise ValueError(f"Unknown goal_source: {cfg.goal_source}")

        self.cfg = cfg
        self.context = context
        self.env = context.env
        self.device = self._resolve_device(cfg.device)
        self.markers = []
        self.info: Dict = {}

        self.diffusion_cfg = self._load_diffusion_config()
        self.diffusion_algorithm = self._load_diffusion_algorithm()
        self.diffusion_algorithm.model.eval()
        for param in self.diffusion_algorithm.model.parameters():
            param.requires_grad_(False)

        dataset_cfg = self.diffusion_cfg.training.dataset
        model_cfg = self.diffusion_cfg.algorithm.config.model
        self.model_horizon = int(model_cfg.horizon)
        self.command_dim = int(model_cfg.command_dim)
        self.execution_horizon = min(int(cfg.execution_horizon), self.model_horizon)
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        self.steps_per_command = max(1, int(round(float(cfg.command_dt) * float(cfg.main_loop_hz))))

        self.obs_cfg = ResidualObservationConfig(
            crop_size=int(dataset_cfg.crop_size),
            goal_scale=float(dataset_cfg.goal_scale),
            velocity_scale=float(dataset_cfg.velocity_scale),
            uncertainty_log_scale=float(dataset_cfg.uncertainty_log_scale),
            uncertainty_max=float(dataset_cfg.uncertainty_max),
            model_horizon=self.model_horizon,
            command_dim=self.command_dim,
        )
        self.obs_builder = ResidualObservationBuilder(self.obs_cfg, device=self.device)

        self.mixer_cfg = CommandMixerConfig(
            residual_horizon=self.execution_horizon,
            mix_mode=str(cfg.mix_mode),
            residual_vxy_scale=float(cfg.residual_vxy_scale),
            residual_heading_scale=float(cfg.residual_heading_scale),
            residual_vel_clip=None if cfg.residual_vel_clip is None else float(cfg.residual_vel_clip),
            residual_heading_clip=None if cfg.residual_heading_clip is None else float(cfg.residual_heading_clip),
            min_velocity_scale=float(cfg.min_velocity_scale),
            max_velocity_scale=float(cfg.max_velocity_scale),
            max_forward_speed=float(cfg.max_forward_speed),
            max_lateral_speed=float(cfg.max_lateral_speed),
        )
        self.residual_cfg = self._load_residual_config()
        self.residual_actor = self._load_residual_actor()
        self.residual_actor.eval()

        self.goal_xy = torch.zeros((self.env.num_envs, 2), dtype=torch.float32, device=self.env.device)
        self.user_goal_xy: Optional[torch.Tensor] = None
        self.command_sequence = torch.zeros(
            (self.env.num_envs, self.execution_horizon, 3),
            dtype=torch.float32,
            device=self.env.device,
        )
        self.residual_action = torch.zeros(
            (self.env.num_envs, self.execution_horizon * 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.sequence_valid = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.sequence_index = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)
        self.command_step_index = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.env.device)

        self.logger.info(
            "Loaded residual diffusion command generator: model_horizon={} execution_horizon={} steps_per_command={}",
            self.model_horizon,
            self.execution_horizon,
            self.steps_per_command,
        )

    def set_goal_xy(self, goal_xy: torch.Tensor) -> None:
        goal_xy = torch.as_tensor(goal_xy, dtype=torch.float32, device=self.env.device)
        if goal_xy.dim() == 1:
            goal_xy = goal_xy.view(1, 2).expand(self.env.num_envs, -1)
        if goal_xy.shape != (self.env.num_envs, 2):
            raise ValueError(f"goal_xy must have shape ({self.env.num_envs}, 2), got {tuple(goal_xy.shape)}")
        self.user_goal_xy = goal_xy.clone()
        self.goal_xy.copy_(goal_xy)
        self._invalidate_sequence()

    def clear_user_goal(self) -> None:
        self.user_goal_xy = None
        self._refresh_goals(torch.arange(self.env.num_envs, device=self.env.device), resample=True)
        self._invalidate_sequence()

    def init_state(self, **kwargs):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self._refresh_goals(env_ids, resample=True)
        self._invalidate_sequence()

    @torch.no_grad()
    def compute_vel_cmd(self, **kwargs):
        reset_buf = getattr(self.env, "reset_buf", None)
        if reset_buf is not None:
            reset_ids = reset_buf.to(device=self.env.device).bool().nonzero(as_tuple=False).flatten()
            if reset_ids.numel() > 0:
                self._refresh_goals(reset_ids, resample=self.cfg.resample_goal_flag)
                self._invalidate_sequence(reset_ids)

        current_xy = self.env.root_local_pos[:, :2]
        distance_to_goal = torch.norm(current_xy - self.goal_xy, dim=-1)
        reached_goal = distance_to_goal <= float(self.cfg.goal_tolerance)
        reached_ids = reached_goal.nonzero(as_tuple=False).flatten()
        if reached_ids.numel() > 0:
            if self.cfg.reset_env_flag:
                if hasattr(self.env, "task_reset"):
                    result = self.env.task_reset(reached_ids, resample_goal=self.cfg.resample_goal_flag)
                    if result.goal_xy is not None:
                        self.goal_xy[reached_ids] = result.goal_xy.to(device=self.env.device, dtype=torch.float32)
                else:
                    self.env.reset_idx(reached_ids)
                    self._refresh_goals(reached_ids, resample=self.cfg.resample_goal_flag)
            else:
                self._refresh_goals(reached_ids, resample=self.cfg.resample_goal_flag)
            self._invalidate_sequence(reached_ids)

        need_sequence = (~self.sequence_valid) | (self.sequence_index >= self.execution_horizon)
        need_ids = need_sequence.nonzero(as_tuple=False).flatten()
        sampled_sequence = False
        if need_ids.numel() > 0:
            self._sample_command_sequence(need_ids)
            sampled_sequence = True

        env_cmd = self._current_env_commands()
        self._advance_sequence_cursor()
        self._update_markers(current_xy[0])
        current_grid_xy = self._current_grid_xy(env_id=0)

        info_struct = ResidualDiffusionPolicyRunningInfo(
            current_xy=current_xy[0].detach().cpu(),
            current_grid_xy=current_grid_xy,
            goal_xy=self.goal_xy[0].detach().cpu(),
            sequence_index=int(self.sequence_index[0].item()),
            command_step_index=int(self.command_step_index[0].item()),
            sampled_sequence=sampled_sequence,
            reached_goal_flag=bool(reached_goal[0].item()),
            traj_cost=0.0,
            reached_goal_count=int(reached_goal.sum().item()),
            mean_distance_to_goal=float(distance_to_goal.mean().detach().cpu()),
            mean_residual_norm=float(self.residual_action.view(self.env.num_envs, -1).norm(dim=-1).mean().detach().cpu()),
            execution_horizon=self.execution_horizon,
            steps_per_command=self.steps_per_command,
        )
        self.info = asdict(info_struct)
        self.info["goal_source"] = self.cfg.goal_source
        return env_cmd

    def retrieve_markers(self) -> List:
        return self.markers

    def get_info(self) -> Dict:
        return self.info

    def _load_diffusion_config(self):
        run_dir = Path(self.cfg.diffusion_run_dir).expanduser()
        config_path = Path(self.cfg.diffusion_config).expanduser() if self.cfg.diffusion_config else run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Diffusion config not found: {config_path}")
        return OmegaConf.load(config_path)

    def _load_diffusion_algorithm(self):
        run_dir = Path(self.cfg.diffusion_run_dir).expanduser()
        checkpoint_path = (
            Path(self.cfg.diffusion_checkpoint).expanduser()
            if self.cfg.diffusion_checkpoint
            else run_dir / "latest.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint_path}")
        algorithm = build_algorithm(self.diffusion_cfg.algorithm).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        algorithm_state = checkpoint.get("algorithm", checkpoint)
        if "model" not in algorithm_state:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain algorithm.model state")
        algorithm.model.load_state_dict(algorithm_state["model"])
        return algorithm

    def _load_residual_config(self):
        run_dir = Path(self.cfg.residual_run_dir).expanduser()
        config_path = Path(self.cfg.residual_config).expanduser() if self.cfg.residual_config else run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Residual config not found: {config_path}")
        return OmegaConf.load(config_path)

    def _load_residual_actor(self):
        run_dir = Path(self.cfg.residual_run_dir).expanduser()
        checkpoint_path = (
            Path(self.cfg.residual_checkpoint).expanduser()
            if self.cfg.residual_checkpoint
            else self._find_default_residual_checkpoint(run_dir)
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Residual checkpoint not found: {checkpoint_path}")

        encoder_output_dim = int(self.diffusion_algorithm.model.cfg.encoder.output_dim)
        num_obs = self.obs_builder.observation_dim(self.context.neural_map, encoder_output_dim)
        num_actions = self.execution_horizon * 3
        policy_cfg = OmegaConf.to_container(self.residual_cfg.ppo.policy, resolve=True)
        actor = ActorCritic(num_obs, num_obs, num_actions, **policy_cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        actor.load_state_dict(state_dict)
        self.logger.info("Loaded residual policy checkpoint={}", checkpoint_path)
        return actor

    def _find_default_residual_checkpoint(self, run_dir: Path) -> Path:
        latest_path = run_dir / "latest.pt"
        if latest_path.exists():
            return latest_path
        model_paths = sorted(run_dir.glob("model_*.pt"), key=self._model_step)
        if not model_paths:
            raise FileNotFoundError(f"No residual checkpoint found in {run_dir}")
        return model_paths[-1]

    @staticmethod
    def _model_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1

    @torch.no_grad()
    def _sample_command_sequence(self, env_ids: torch.Tensor) -> None:
        batch = self.obs_builder.build_batch(self.env, self.context.neural_map, self.goal_xy)
        if hasattr(self.diffusion_algorithm, "infer"):
            target = self.diffusion_algorithm.infer(batch, self.model_horizon)
        elif hasattr(self.diffusion_algorithm, "sample"):
            target = self.diffusion_algorithm.sample(batch, self.model_horizon)
        else:
            target = self.diffusion_algorithm.model(batch)
        base_commands = training_target_to_commands(target, velocity_scale=float(self.obs_cfg.velocity_scale))
        base_commands[..., 2] = torch.remainder(base_commands[..., 2] + math.pi, 2.0 * math.pi) - math.pi
        base_commands = base_commands[:, : self.execution_horizon].to(self.device)

        residual_obs = self.obs_builder.build_observation(
            self.env,
            self.context.neural_map,
            self.goal_xy,
            self._encode_condition,
            batch=batch,
        )
        self.residual_action = self.residual_actor.act_inference(residual_obs)
        mixed = mix_residual_commands(base_commands, self.residual_action, self.mixer_cfg)

        self.command_sequence[env_ids] = mixed.to(device=self.env.device, dtype=self.command_sequence.dtype)[env_ids]
        self.sequence_index[env_ids] = 0
        self.command_step_index[env_ids] = 0
        self.sequence_valid[env_ids] = True

    def _encode_condition(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model = self.diffusion_algorithm.model
        if hasattr(model, "encode_condition"):
            return model.encode_condition(batch)
        if hasattr(model, "encoder"):
            return model.encoder(batch)
        raise TypeError("Diffusion model does not expose encode_condition(...) or encoder(...)")

    def _current_env_commands(self) -> torch.Tensor:
        env_cmd = torch.zeros_like(self.env.commands)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        seq_idx = torch.clamp(self.sequence_index, min=0, max=self.execution_horizon - 1)
        cmd = self.command_sequence[env_ids, seq_idx]
        env_cmd[:, 0:2] = cmd[:, 0:2]
        env_cmd[:, 3] = cmd[:, 2]
        if env_cmd.shape[1] > 2 and hasattr(self.env, "yaw"):
            heading_error = wrap_pi(env_cmd[:, 3] - self.env.yaw)
            env_cmd[:, 2] = torch.clamp(0.5 * heading_error, min=-1.0, max=1.0)
        return env_cmd

    def _advance_sequence_cursor(self) -> None:
        self.command_step_index += 1
        advance = self.command_step_index >= self.steps_per_command
        self.command_step_index[advance] = 0
        self.sequence_index[advance] += 1

    def _refresh_goals(self, env_ids: torch.Tensor, *, resample: bool) -> None:
        if env_ids.numel() == 0:
            return
        if self.user_goal_xy is not None:
            self.goal_xy[env_ids] = self.user_goal_xy[env_ids]
            return
        if resample:
            env_ids_env = env_ids.to(device=self.env.device)
            if hasattr(self.env, "resample_goals"):
                result = self.env.resample_goals(env_ids_env)
                assert result.goal_xy is not None
                sampled = result.goal_xy
            else:
                sampled = self.env.sample_goals(env_ids_env)[:, :2]
            self.goal_xy[env_ids] = sampled.to(device=self.env.device, dtype=torch.float32)

    def _invalidate_sequence(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.sequence_valid[env_ids] = False
        self.sequence_index[env_ids] = 0
        self.command_step_index[env_ids] = 0

    def _current_grid_xy(self, env_id: int = 0) -> tuple[int, int]:
        neural_map = self.context.neural_map
        grid_xy = self.obs_builder.xy_to_grid(neural_map, self.env.root_local_pos[:, :2])
        x_idx = int(grid_xy[env_id, 0].item())
        y_idx = int(grid_xy[env_id, 1].item())
        return (x_idx, y_idx)

    def _resolve_device(self, requested: str) -> torch.device:
        if requested in ("gpu", "cuda"):
            requested = "cuda"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("Requested {} but CUDA is unavailable; using CPU", requested)
            return torch.device("cpu")
        return torch.device(requested)

    def _update_markers(self, current_xy: torch.Tensor) -> None:
        if self.env.num_envs <= 0:
            self.markers = []
            return
        terrain_id = self.env.env_terrain_map[0]
        terrain_origin = self.env.terrain_loader._instances[terrain_id].array_offset
        robot_marker = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
        goal_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 1))
        robot_h = self.env.terrain_loader.get_height(terrain_id, current_xy)
        goal_h = self.env.terrain_loader.get_height(terrain_id, self.goal_xy[0])
        robot_pose = gymapi.Transform(
            gymapi.Vec3(current_xy[0].item() + terrain_origin[0], current_xy[1].item() + terrain_origin[1], robot_h),
            r=None,
        )
        goal_pose = gymapi.Transform(
            gymapi.Vec3(self.goal_xy[0, 0].item() + terrain_origin[0], self.goal_xy[0, 1].item() + terrain_origin[1], goal_h),
            r=None,
        )
        self.markers = [(robot_marker, robot_pose), (goal_marker, goal_pose)]
