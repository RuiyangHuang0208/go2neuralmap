from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from isaacgym import gymapi, gymutil
from loguru import logger
from omegaconf import OmegaConf
import torch

from rsl_rl.runners import OnPolicyRunner
from utils.rl_nav_policy.command import scale_rl_nav_actions
from utils.rl_nav_policy.env import RLNavEnvConfig
from utils.rl_nav_policy.observation import RLNavObservationBuilder

from .navigation_task_evaluator import NavigationTaskEvaluator, NavigationTaskEvaluatorConfig
from .vel_command_generator import VelCommandGenerator, VelCommandGeneratorConfig


def _merge_config(cfg_obj, cfg_dict):
    schema = OmegaConf.structured(cfg_obj)
    merged_cfg = OmegaConf.merge(schema, cfg_dict)
    return OmegaConf.to_object(merged_cfg)


@dataclass
class RLNavPolicyVelCommandGeneratorConfig(VelCommandGeneratorConfig):
    run_dir: str = "logs/rl_nav_policy"
    checkpoint: Optional[str] = None
    config: Optional[str] = None
    device: str = "cuda"
    command_dt: float = 0.25
    main_loop_hz: float = 50.0
    goal_tolerance: Optional[float] = None
    resample_goal_flag: bool = True
    reset_env_flag: bool = False
    goal_source: str = "sampled"
    goal_xy_list: Optional[List[List[float]]] = None
    goal_list_start_index: int = 0
    timeout_limit_s: Optional[float] = None
    timeout_limit_steps: Optional[int] = None
    timeout_action: str = "task_reset"
    advance_goal_on_success: bool = True
    advance_goal_on_timeout: bool = True
    advance_goal_on_done: bool = True
    draw_goal_marker: bool = True
    goal_marker_radius: float = 0.15
    visualize_attention_weights: bool = False
    attention_marker_min_radius: float = 0.015
    attention_marker_max_radius: float = 0.08
    attention_marker_height_offset: float = 0.08
    attention_marker_height_scale: float = 0.12
    disable_low_level_command_resampling: bool = True
    rl_nav_overrides: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class RLNavPolicyRunningInfo:
    current_xy: Optional[torch.Tensor] = None
    current_grid_xy: Optional[tuple[int, int]] = None
    goal_xy: Optional[torch.Tensor] = None
    distance_to_goal: Optional[torch.Tensor] = None
    reached_goal_flag: bool = False
    reached_goal_count: int = 0
    traj_cost: float = 0.0
    action: Optional[torch.Tensor] = None
    command: Optional[torch.Tensor] = None


class _RLNavInferenceEnvAdapter:
    def __init__(self, *, num_envs: int, num_obs: int, num_actions: int, device: torch.device):
        self.num_envs = int(num_envs)
        self.num_obs = int(num_obs)
        self.num_actions = int(num_actions)
        self.num_privileged_obs = None
        self.max_episode_length = 1
        self.device = device

    def reset(self):
        obs = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        return obs, None


class RLNavPolicyVelCommandGenerator(VelCommandGenerator):
    def __init__(self, cfg: RLNavPolicyVelCommandGeneratorConfig, context=None, **kwargs):
        self.logger = logger.bind(module_name="VelCmdGen.RLNavPolicy")
        if context is None:
            raise ValueError("No context is passed")
        if cfg.goal_source not in ("sampled", "user", "list"):
            raise ValueError(f"Unknown goal_source: {cfg.goal_source}")

        self.cfg = cfg
        self.context = context
        self.env = context.env
        self.neural_map = context.neural_map
        self.device = self._resolve_device(cfg.device)
        self.markers: List = []
        self.info: Dict = {}
        self.attention_grid: Optional[torch.Tensor] = None
        self.actor_critic = None

        self.run_dir = Path(cfg.run_dir).expanduser()
        self.config_path = Path(cfg.config).expanduser() if cfg.config is not None else self.run_dir / "config.yaml"
        self.checkpoint_path = (
            Path(cfg.checkpoint).expanduser() if cfg.checkpoint is not None else self._latest_checkpoint(self.run_dir)
        )

        self.cfg_omega = self._load_config()
        self.rl_nav_cfg = self._build_rl_nav_cfg()
        self._configure_env_for_yaw_rate_commands()
        self.obs_builder = RLNavObservationBuilder(self.rl_nav_cfg.observation, device=self.device)
        self.observation_mode = str(self.rl_nav_cfg.observation_mode)
        if self.observation_mode == "map":
            self.num_obs = self.obs_builder.observation_dim(self.neural_map)
        elif self.observation_mode == "condition_only":
            self.num_obs = self.obs_builder.condition_observation_dim()
        else:
            raise ValueError("rl_nav.observation_mode must be 'map' or 'condition_only'")
        self.current_command = torch.zeros((self.env.num_envs, 3), dtype=torch.float32, device=self.device)
        self.current_action = torch.zeros((self.env.num_envs, 3), dtype=torch.float32, device=self.device)
        self.steps_per_command = max(1, int(round(float(cfg.command_dt) * float(cfg.main_loop_hz))))
        self.command_step_index = 0

        self.policy = self._load_policy()
        task_cfg = NavigationTaskEvaluatorConfig(
            goal_tolerance=cfg.goal_tolerance,
            goal_source=cfg.goal_source,
            goal_xy_list=cfg.goal_xy_list,
            goal_list_start_index=cfg.goal_list_start_index,
            timeout_limit_s=cfg.timeout_limit_s,
            timeout_limit_steps=cfg.timeout_limit_steps,
            timeout_action=cfg.timeout_action,
            advance_goal_on_success=cfg.advance_goal_on_success,
            advance_goal_on_timeout=cfg.advance_goal_on_timeout,
            advance_goal_on_done=cfg.advance_goal_on_done,
        )
        self.task_evaluator = NavigationTaskEvaluator(
            env=self.env,
            cfg=task_cfg,
            default_goal_tolerance=float(self.rl_nav_cfg.reward.goal_tolerance),
        )
        self.goal_xy = self.task_evaluator.goal_xy
        self.last_grid_xy_by_env: List[Optional[tuple[int, int]]] = [None for _ in range(int(self.env.num_envs))]
        self.traj_cost_by_env: List[float] = [0.0 for _ in range(int(self.env.num_envs))]

        self.logger.info(
            "Loaded RL nav policy checkpoint={} num_obs={} num_envs={} steps_per_command={}",
            self.checkpoint_path,
            self.num_obs,
            self.env.num_envs,
            self.steps_per_command,
        )

    def set_goal_xy(self, goal_xy: torch.Tensor) -> None:
        self.task_evaluator.set_goal_xy(goal_xy)

    def clear_user_goal(self) -> None:
        current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
        self.task_evaluator.clear_user_goal(current_xy)

    def init_state(self, **kwargs):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self._configure_env_for_yaw_rate_commands()
        current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
        self.task_evaluator.init_tasks(current_xy, resample=True)
        self.current_command.zero_()
        self.current_action.zero_()
        self.command_step_index = 0
        self.last_grid_xy_by_env = [self._current_grid_xy(env_id) for env_id in range(int(self.env.num_envs))]
        self.traj_cost_by_env = [0.0 for _ in range(int(self.env.num_envs))]
        self.info = {}
        self.attention_grid = None

    @torch.no_grad()
    def compute_vel_cmd(self, **kwargs):
        self._configure_env_for_yaw_rate_commands()
        reset_buf = getattr(self.env, "reset_buf", None)
        current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
        reset_ids = self.task_evaluator.handle_external_resets(reset_buf, current_xy)
        if reset_ids.numel() > 0:
            self.current_command[reset_ids.to(device=self.device)] = 0.0
            self.current_action[reset_ids.to(device=self.device)] = 0.0
            self._reset_trajectory_state(reset_ids)
            self.command_step_index = 0

        current_grid_xy_by_env = self._update_grid_traj_costs()
        task_update = self.task_evaluator.update(
            current_xy=current_xy,
            dt=float(getattr(self.env, "dt", 0.0)),
        )
        terminal_ids = task_update.terminal_ids
        if terminal_ids.numel() > 0:
            timeout_reset_ids = (
                task_update.timeout_ids
                if self.cfg.timeout_action == "env_reset"
                else torch.zeros(0, dtype=torch.long, device=self.env.device)
            )
            success_reset_ids = (
                task_update.success_ids
                if self.cfg.reset_env_flag
                else torch.zeros(0, dtype=torch.long, device=self.env.device)
            )
            reset_terminal_ids = torch.unique(torch.cat((timeout_reset_ids, success_reset_ids)))
            if reset_terminal_ids.numel() > 0:
                self._reset_envs(reset_terminal_ids)
                current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
            self.task_evaluator.complete_terminals(task_update, current_xy=current_xy)
            self.current_command[terminal_ids.to(device=self.device)] = 0.0
            self.current_action[terminal_ids.to(device=self.device)] = 0.0
            self._reset_trajectory_state(terminal_ids)
            self.command_step_index = 0

        sampled_action = False
        if self.command_step_index <= 0:
            obs = self._build_policy_observation()
            obs = obs.to(device=self.device, dtype=torch.float32)
            actions = self.policy(obs)
            if (
                self.observation_mode == "map"
                and bool(self.cfg.visualize_attention_weights)
                and self.actor_critic is not None
                and hasattr(self.actor_critic, "get_actor_attention_grid")
            ):
                self.attention_grid = self.actor_critic.get_actor_attention_grid(obs).detach()
            commands, command_info = scale_rl_nav_actions(actions, self.rl_nav_cfg.command, return_info=True)
            self.current_action = actions.detach().clone()
            self.current_command = commands.detach().clone()
            sampled_action = True
        else:
            actions = self.current_action
            commands = self.current_command
            _, command_info = scale_rl_nav_actions(actions, self.rl_nav_cfg.command, return_info=True)

        self.command_step_index = (self.command_step_index + 1) % self.steps_per_command

        env_cmd = torch.zeros_like(self.env.commands)
        env_cmd[:, :3] = commands.to(device=self.env.device, dtype=self.env.commands.dtype)
        if env_cmd.shape[1] > 3:
            env_cmd[:, 3] = 0.0

        self._update_markers()
        info_struct = RLNavPolicyRunningInfo(
            current_xy=current_xy[0].detach().cpu(),
            goal_xy=self.goal_xy[0].detach().cpu(),
            distance_to_goal=torch.norm(current_xy - self.goal_xy, dim=-1).detach().cpu(),
            reached_goal_flag=bool(task_update.reached_goal[0].item()),
            reached_goal_count=int(self.task_evaluator.reached_goal_count[0].item()),
            action=actions[0].detach().cpu(),
            command=commands[0].detach().cpu(),
            current_grid_xy=current_grid_xy_by_env[0] if current_grid_xy_by_env else None,
            traj_cost=self.traj_cost_by_env[0] if self.traj_cost_by_env else 0.0,
        )
        self.info = asdict(info_struct)
        self.info.update(self.task_evaluator.info_for_env(task_update, 0))
        self.info["goal_source"] = self.cfg.goal_source
        self.info["command_speed"] = command_info["command_speed"].detach().cpu()
        self.info["yaw_rate_abs"] = command_info["yaw_rate_abs"].detach().cpu()
        self.info["sampled_action"] = sampled_action
        self.info["command_step_index"] = self.command_step_index
        self.info["steps_per_command"] = self.steps_per_command
        if self.attention_grid is not None and self.attention_grid.numel() > 0:
            attention_env0 = self.attention_grid[0].detach().cpu()
            flat_attention = attention_env0.reshape(-1)
            entropy = -(flat_attention * torch.log(torch.clamp(flat_attention, min=1.0e-8))).sum()
            self.info["attention_grid"] = attention_env0
            self.info["attention_max"] = float(flat_attention.max().item())
            self.info["attention_entropy"] = float(entropy.item())
        return env_cmd

    def _build_policy_observation(self) -> torch.Tensor:
        if self.observation_mode == "map":
            return self.obs_builder.build_observation(
                self.env,
                self.neural_map,
                self.goal_xy.to(device=self.device),
                previous_command=self.current_command,
            )
        if self.observation_mode == "condition_only":
            return self.obs_builder.build_condition_observation(
                self.env,
                self.goal_xy.to(device=self.device),
                previous_command=self.current_command,
            )
        raise ValueError("rl_nav.observation_mode must be 'map' or 'condition_only'")

    def retrieve_markers(self) -> List:
        return self.markers

    def get_info(self) -> Dict:
        return self.info

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"RL nav config not found: {self.config_path}")
        return OmegaConf.load(self.config_path)

    def _build_rl_nav_cfg(self) -> RLNavEnvConfig:
        cfg = _merge_config(RLNavEnvConfig(), self.cfg_omega.rl_nav)
        if self.cfg.rl_nav_overrides:
            cfg = _merge_config(cfg, OmegaConf.create(self.cfg.rl_nav_overrides))
        return cfg

    def _load_policy(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"RL nav checkpoint not found: {self.checkpoint_path}")
        train_cfg = OmegaConf.to_container(self.cfg_omega.ppo, resolve=True)
        train_cfg["runner"]["logger"] = "none"
        adapter = _RLNavInferenceEnvAdapter(
            num_envs=self.env.num_envs,
            num_obs=self.num_obs,
            num_actions=3,
            device=self.device,
        )
        runner = OnPolicyRunner(adapter, train_cfg, log_dir=None, device=str(self.device))
        loaded_dict = torch.load(str(self.checkpoint_path), map_location=self.device)
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        runner.current_learning_iteration = loaded_dict.get("iter", 0)
        self.actor_critic = runner.alg.actor_critic
        return runner.get_inference_policy(device=self.device)

    def _resolve_device(self, requested: str) -> torch.device:
        if requested in ("gpu", "cuda"):
            requested = "cuda"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("Requested {} but CUDA is unavailable; using CPU", requested)
            return torch.device("cpu")
        return torch.device(requested)

    def _configure_env_for_yaw_rate_commands(self) -> None:
        commands_cfg = getattr(getattr(self.env, "cfg", None), "commands", None)
        if commands_cfg is None:
            return
        if getattr(commands_cfg, "heading_command", False):
            self.logger.warning("Forcing env.cfg.commands.heading_command=false for RL nav yaw-rate commands.")
        commands_cfg.heading_command = False
        if bool(self.cfg.disable_low_level_command_resampling):
            commands_cfg.resampling_time = 1.0e9

    def _refresh_goals(self, env_ids: torch.Tensor, *, resample: bool) -> None:
        self.task_evaluator.refresh_goals(env_ids, resample=resample)

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        if hasattr(self.env, "task_reset"):
            self.env.task_reset(env_ids, resample_goal=False)
        else:
            self.env.reset_idx(env_ids)
        self._reset_trajectory_state(env_ids)

    def _sync_env_task_goals(self, env_ids: torch.Tensor, goal_xy: torch.Tensor) -> None:
        self.task_evaluator._sync_env_task_goals(env_ids, goal_xy)

    def _current_grid_xy(self, env_id: int = 0) -> tuple[int, int]:
        grid_xy = self.obs_builder.xy_to_grid(self.neural_map, self.env.root_local_pos[:, :2])
        x_idx = int(grid_xy[env_id, 0].item())
        y_idx = int(grid_xy[env_id, 1].item())
        return (x_idx, y_idx)

    def _update_grid_traj_costs(self) -> List[tuple[int, int]]:
        if len(self.last_grid_xy_by_env) != int(self.env.num_envs):
            self.last_grid_xy_by_env = [None for _ in range(int(self.env.num_envs))]
            self.traj_cost_by_env = [0.0 for _ in range(int(self.env.num_envs))]

        current_grid_xy_by_env = [self._current_grid_xy(env_id) for env_id in range(int(self.env.num_envs))]
        for env_id, current_grid_xy in enumerate(current_grid_xy_by_env):
            last_grid_xy = self.last_grid_xy_by_env[env_id]
            if last_grid_xy is None:
                self.last_grid_xy_by_env[env_id] = current_grid_xy
                continue
            if last_grid_xy == current_grid_xy:
                continue
            if not self._is_cardinal_adjacent(last_grid_xy, current_grid_xy):
                self.last_grid_xy_by_env[env_id] = current_grid_xy
                continue
            if hasattr(self.context, "cost_map") and hasattr(self.context.cost_map, "get_traversal_cost"):
                self.traj_cost_by_env[env_id] += self.context.cost_map.get_traversal_cost(
                    last_grid_xy,
                    current_grid_xy,
                    env_id=env_id,
                )
            self.last_grid_xy_by_env[env_id] = current_grid_xy
        return current_grid_xy_by_env

    @staticmethod
    def _is_cardinal_adjacent(
        previous_grid_xy: tuple[int, int],
        current_grid_xy: tuple[int, int],
    ) -> bool:
        dx = int(current_grid_xy[0]) - int(previous_grid_xy[0])
        dy = int(current_grid_xy[1]) - int(previous_grid_xy[1])
        return abs(dx) + abs(dy) == 1

    def _reset_trajectory_state(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long).flatten()
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            if env_id >= len(self.last_grid_xy_by_env):
                continue
            self.last_grid_xy_by_env[env_id] = self._current_grid_xy(env_id)
            self.traj_cost_by_env[env_id] = 0.0

    def _update_markers(self) -> None:
        self.markers = []
        if self.env.num_envs <= 0:
            return
        if not hasattr(self.env, "terrain_loader") or not hasattr(self.env, "env_terrain_map"):
            return
        terrain_id = self.env.env_terrain_map[0]
        terrain_idx = int(terrain_id.detach().cpu().item()) if isinstance(terrain_id, torch.Tensor) else int(terrain_id)
        terrain_origin = self.env.terrain_loader._instances[terrain_idx].array_offset
        if bool(self.cfg.draw_goal_marker):
            goal_h = self.env.terrain_loader.get_height(terrain_id, self.goal_xy[0])
            goal_marker = gymutil.WireframeSphereGeometry(
                float(self.cfg.goal_marker_radius),
                8,
                8,
                None,
                color=(1.0, 0.0, 0.0),
            )
            goal_pose = gymapi.Transform(
                gymapi.Vec3(
                    self.goal_xy[0, 0].item() + terrain_origin[0],
                    self.goal_xy[0, 1].item() + terrain_origin[1],
                    float(torch.as_tensor(goal_h).detach().cpu().item()) + float(self.cfg.goal_marker_radius),
                ),
                r=None,
            )
            self.markers.append((goal_marker, goal_pose))
        if bool(self.cfg.visualize_attention_weights):
            self.markers.extend(
                self._build_attention_weight_markers(
                    terrain_id=terrain_id,
                    terrain_origin=terrain_origin,
                )
            )

    def _build_attention_weight_markers(self, *, terrain_id, terrain_origin) -> List:
        if self.attention_grid is None or self.attention_grid.numel() == 0:
            return []
        if self.env.num_envs <= 0:
            return []

        current_xy = self.env.root_local_pos[:, :2].to(device=self.device, dtype=torch.float32)
        start_grid = self.obs_builder.xy_to_grid(self.neural_map, current_xy).to(device=self.device)
        _, grid_x, grid_y, valid = self.obs_builder.crop_indices_batched(self.neural_map, start_grid)
        cell_xy = self.obs_builder.crop_cell_xy_batched(self.neural_map, grid_x, grid_y)[0]
        valid_flat = valid[0].reshape(-1)
        weights = self.attention_grid[0].to(device=self.device, dtype=torch.float32).reshape(-1)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        if weights.numel() != cell_xy.shape[0]:
            self.logger.warning(
                "Attention grid size {} does not match local crop size {}; skipping attention markers.",
                weights.numel(),
                cell_xy.shape[0],
            )
            return []

        max_weight = (
            torch.clamp(weights[valid_flat].max(), min=1.0e-8)
            if torch.any(valid_flat)
            else torch.tensor(1.0, dtype=torch.float32, device=self.device)
        )
        normalized = torch.clamp(weights / max_weight, min=0.0, max=1.0)
        markers = []
        for idx in range(cell_xy.shape[0]):
            if not bool(valid_flat[idx].item()):
                continue
            strength = float(normalized[idx].detach().cpu().item())
            radius = self._attention_marker_radius(strength)
            color = self._attention_marker_color(strength)
            pos_xy = cell_xy[idx].to(device=self.env.device, dtype=torch.float32)
            marker_h = self.env.terrain_loader.get_height(terrain_id, pos_xy)
            marker = gymutil.WireframeSphereGeometry(
                radius,
                4,
                4,
                None,
                color=color,
            )
            marker_pose = gymapi.Transform(
                gymapi.Vec3(
                    pos_xy[0].item() + terrain_origin[0],
                    pos_xy[1].item() + terrain_origin[1],
                    float(torch.as_tensor(marker_h).detach().cpu().item())
                    + float(self.cfg.attention_marker_height_offset)
                    + strength * float(self.cfg.attention_marker_height_scale),
                ),
                r=None,
            )
            markers.append((marker, marker_pose))
        return markers

    def _attention_marker_radius(self, strength: float) -> float:
        min_radius = float(self.cfg.attention_marker_min_radius)
        max_radius = float(self.cfg.attention_marker_max_radius)
        return min_radius + float(strength) * max(0.0, max_radius - min_radius)

    @staticmethod
    def _attention_marker_color(strength: float) -> tuple[float, float, float]:
        strength = max(0.0, min(1.0, float(strength)))
        return (
            0.15 + 0.85 * strength,
            0.20 + 0.55 * (1.0 - abs(strength - 0.5) * 2.0),
            1.0 - 0.85 * strength,
        )

    @staticmethod
    def _latest_checkpoint(run_dir: Path) -> Path:
        paths = sorted(run_dir.glob("model_*.pt"), key=RLNavPolicyVelCommandGenerator._model_step)
        if not paths:
            raise FileNotFoundError(f"No model_*.pt checkpoint found in {run_dir}")
        return paths[-1]

    @staticmethod
    def _model_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[-1])
        except ValueError:
            return -1
