from __future__ import annotations

from dataclasses import dataclass, field
import time
import re
from typing import Dict, Optional

import torch

from rsl_rl.env import VecEnv

from .command import RLNavCommandConfig, scale_rl_nav_actions
from .observation import RLNavObservationBuilder, RLNavObservationConfig


@dataclass
class RLNavCollisionUncertaintyScaleConfig:
    enabled: bool = False
    scope: str = "crop_mean"
    min_scale: float = 0.75
    max_scale: float = 1.5


@dataclass
class RLNavRepeatedCollisionPenaltyConfig:
    enabled: bool = False
    penalty_scale: float = 1.0
    radius_cells: int = 1
    max_count: float = 3.0
    decay: float = 1.0


@dataclass
class RLNavRewardConfig:
    variant: str = "shaped"
    goal_tolerance: float = 0.35
    reached_goal_reward: float = 10.0
    middle_point_tolerance: float = 0.35
    middle_point_reward: float = 0.0
    collision_penalty: float = 2.0
    termination_penalty: float = 10.0
    progress_scale: float = 2.0
    progress_positive_only: bool = True
    time_penalty: float = 0.01
    command_penalty_scale: float = 0.01
    command_smoothness_penalty_scale: float = 0.02
    collision_uncertainty_scale: RLNavCollisionUncertaintyScaleConfig = field(
        default_factory=RLNavCollisionUncertaintyScaleConfig
    )
    repeated_collision_penalty: RLNavRepeatedCollisionPenaltyConfig = field(
        default_factory=RLNavRepeatedCollisionPenaltyConfig
    )


@dataclass
class RLNavGoalSamplingConfig:
    min_distance: float = 0.0
    max_distance: Optional[float] = None
    max_attempts: int = 20
    fallback: str = "accept_last"
    avoid_previous: bool = False
    previous_tolerance: float = 1.0e-4


@dataclass
class RLNavTerrainResamplingConfig:
    on_task_success: bool = False
    on_map_reset: bool = False
    on_timeout: bool = False
    on_collision_done: bool = False


@dataclass
class RLNavEnvConfig:
    observation: RLNavObservationConfig = field(default_factory=RLNavObservationConfig)
    command: RLNavCommandConfig = field(default_factory=RLNavCommandConfig)
    reward: RLNavRewardConfig = field(default_factory=RLNavRewardConfig)
    goal_sampling: RLNavGoalSamplingConfig = field(default_factory=RLNavGoalSamplingConfig)
    terrain_resampling: RLNavTerrainResamplingConfig = field(default_factory=RLNavTerrainResamplingConfig)
    command_dt: float = 0.5
    main_loop_hz: float = 50.0
    max_episode_steps: int = 80
    collision_done_threshold: Optional[int] = None
    map_reset_success_threshold: int = 5
    resample_task_on_timeout: bool = True
    resample_task_on_collision_done: bool = True
    disable_low_level_command_resampling: bool = True
    terminate_on_map_out_of_bounds: bool = True
    profiling_enabled: bool = True
    debug_observation_nan: bool = False
    observation_mode: str = "map"


class RLNavPolicyEnv(VecEnv):
    """High-level RL navigation wrapper using Neural Map observations."""

    def __init__(
        self,
        *,
        env,
        neural_map,
        locomotion_model,
        cfg: RLNavEnvConfig,
        device: torch.device | str,
    ):
        self.env = env
        self.neural_map = neural_map
        self.locomotion_model = locomotion_model
        self.cfg = cfg
        self.device = torch.device(device)

        self.num_envs = int(env.num_envs)
        self.num_actions = 3
        self.max_episode_length = int(cfg.max_episode_steps)
        self.num_privileged_obs = None
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras: Dict = {}

        self.steps_per_command = max(1, int(round(float(cfg.command_dt) * float(cfg.main_loop_hz))))
        self.obs_builder = RLNavObservationBuilder(cfg.observation, device=self.device)
        self.observation_mode = str(cfg.observation_mode)
        if self.observation_mode == "map":
            self.num_obs = self.obs_builder.observation_dim(self.neural_map)
        elif self.observation_mode == "condition_only":
            self.num_obs = self.obs_builder.condition_observation_dim()
        else:
            raise ValueError("rl_nav.observation_mode must be 'map' or 'condition_only'")
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.privileged_obs_buf = None

        self.goal_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.goal_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.middle_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.middle_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.middle_reward_claimed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.task_start_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.repeated_collision_map = self._new_repeated_collision_map()
        self.success_count_since_map_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_command = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.previous_command = torch.zeros_like(self.current_command)
        self._profile: Dict[str, float] = {}

        self._configure_low_level_command_mode()
        self._freeze_models()
        self._store_current_start(torch.arange(self.num_envs, device=self.device))
        self._sample_goals(torch.arange(self.num_envs, device=self.device))
        self._sample_middle_points(torch.arange(self.num_envs, device=self.device))
        self._refresh_observation()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        self._reset_map_episode(env_ids)
        self._refresh_observation()
        return self.obs_buf, self.privileged_obs_buf

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        step_profile_start = self._profile_now()
        self._profile = self._new_profile()
        actions = actions.to(device=self.device, dtype=torch.float32)
        command_start = self._profile_now()
        command, command_info = scale_rl_nav_actions(actions, self.cfg.command, return_info=True)
        self._profile_add("command_s", command_start)
        previous_command = self.current_command.clone()
        self.previous_command = previous_command
        self.current_command = command
        previous_distance = torch.norm(self.goal_xy - self.env.root_local_pos[:, :2].to(self.device), dim=-1)

        any_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        any_low_level_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        any_map_out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        low_level_obs = self.env.get_observations()
        env_cmd = self._commands_to_env(command)
        for _ in range(self.steps_per_command):
            self.env.commands[:, :] = env_cmd
            obs_patch_start = self._profile_now()
            low_level_obs = self._apply_command_to_locomotion_obs(low_level_obs)
            self._profile_add("low_level_obs_patch_s", obs_patch_start)
            locomotion_start = self._profile_now()
            self.locomotion_model.compute_latents(low_level_obs.detach())
            low_level_actions = self.locomotion_model.act()
            self._profile_add("locomotion_s", locomotion_start)
            env_step_start = self._profile_now()
            step_out = self.env.step(low_level_actions.detach())
            self._profile_add("env_step_s", env_step_start)
            normalize_start = self._profile_now()
            low_level_obs, privileged_obs, _, dones, _, _, _ = self._normalize_env_step_output(step_out)
            step_done = dones.to(device=self.device, dtype=torch.bool)
            map_out_of_bounds = self._map_out_of_bounds_mask(self.env.root_local_pos[:, :2])
            any_map_out_of_bounds |= map_out_of_bounds
            self._profile_add("post_step_bookkeeping_s", normalize_start)

            long_latent = self.locomotion_model.out["latent_long"]
            short_latent = self.locomotion_model.out["short_proj"]
            map_update_start = self._profile_now()
            self._update_maps_preserving_done_envs(
                done_mask=step_done | map_out_of_bounds,
                long_latents=long_latent.clone(),
                short_latents=short_latent.clone(),
                xy_coord=self.env.root_local_pos[:, :2],
                root_velocity=self.env.root_states[:, 7:9],
            )
            self._profile_add("map_update_s", map_update_start)
            collision_start = self._profile_now()
            collision_flags = self._collision_flags()
            any_collision |= collision_flags
            any_low_level_done |= step_done
            self._profile_add("collision_check_s", collision_start)
            reset_low_level_ids = (step_done | map_out_of_bounds).nonzero(as_tuple=False).flatten()
            if reset_low_level_ids.numel() > 0:
                low_reset_start = self._profile_now()
                self._reset_robot_to_task_start(reset_low_level_ids)
                low_level_obs = self.env.get_observations()
                self._profile_add("low_level_reset_s", low_reset_start)

        current_distance = torch.norm(self.goal_xy - self.env.root_local_pos[:, :2].to(self.device), dim=-1)
        reached_goal = current_distance <= float(self.cfg.reward.goal_tolerance)
        middle_valid, middle_distance, reached_middle = self._middle_point_status()
        middle_reached_this_task = self.middle_reward_claimed | reached_middle
        self.collision_count += any_collision.long()
        collision_done = torch.zeros_like(reached_goal)
        if self.cfg.collision_done_threshold is not None:
            collision_done = self.collision_count >= int(self.cfg.collision_done_threshold)

        self.episode_length_buf += 1
        timeout = self.episode_length_buf >= int(self.cfg.max_episode_steps)
        map_out_of_bounds_done = any_map_out_of_bounds & bool(self.cfg.terminate_on_map_out_of_bounds)
        dones = reached_goal | any_low_level_done | timeout | collision_done | map_out_of_bounds_done
        collision_penalty_scale = self._collision_penalty_scale()
        repeated_collision_penalty = self._repeated_collision_penalty(any_collision)
        rewards = self._compute_reward(
            command=command,
            previous_command=previous_command,
            previous_distance=previous_distance,
            current_distance=current_distance,
            reached_goal=reached_goal,
            reached_middle=reached_middle,
            collision=any_collision,
            fixed_collision=any_map_out_of_bounds & ~any_collision,
            collision_penalty_scale=collision_penalty_scale,
            repeated_collision_penalty=repeated_collision_penalty,
            terminated=any_low_level_done | collision_done | map_out_of_bounds_done,
        )
        self.middle_reward_claimed |= reached_middle

        self.rew_buf = rewards
        self.reset_buf = dones
        if torch.any(dones):
            done_reset_start = self._profile_now()
            self._handle_done_resets(
                reached_goal=reached_goal,
                low_level_done=any_low_level_done | map_out_of_bounds_done,
                timeout=timeout,
                collision_done=collision_done,
            )
            self._profile_add("done_reset_s", done_reset_start)

        obs_refresh_start = self._profile_now()
        self._refresh_observation()
        self._profile_add("obs_refresh_s", obs_refresh_start)
        self._profile_add("step_total_s", step_profile_start)
        progress = previous_distance - current_distance
        self.extras = {
            "episode": {
                "success_rate": torch.masked_select(reached_goal.float(), dones),
                "collision_rate": torch.masked_select(any_collision.float(), dones),
                "map_out_of_bounds_rate": torch.masked_select(any_map_out_of_bounds.float(), dones),
                "final_distance_to_goal": torch.masked_select(current_distance.detach(), dones),
                "timeout_rate": torch.masked_select(timeout.float(), dones),
                "low_level_done_rate": torch.masked_select(any_low_level_done.float(), dones),
                "collision_done_rate": torch.masked_select(collision_done.float(), dones),
                "middle_reached_rate": torch.masked_select(middle_reached_this_task.float(), dones),
                **self._terrain_episode_metrics(dones, reached_goal),
            },
            "step": {
                "step_reached_goal_rate": reached_goal.float(),
                "step_reached_middle_rate": reached_middle.float(),
                "step_middle_valid_rate": middle_valid.float(),
                "step_middle_distance": middle_distance.detach(),
                "step_collision_rate": any_collision.float(),
                "step_map_out_of_bounds_rate": any_map_out_of_bounds.float(),
                "step_distance_to_goal": current_distance.detach(),
                "step_progress": progress.detach(),
                "step_positive_progress": torch.clamp(progress, min=0.0).detach(),
                "step_command_speed": command_info["command_speed"].detach(),
                "step_yaw_rate_abs": command_info["yaw_rate_abs"].detach(),
                "step_command_delta_norm": torch.norm(command - previous_command, dim=-1).detach(),
                "step_collision_penalty_scale": collision_penalty_scale.detach(),
                "step_repeated_collision_penalty": repeated_collision_penalty.detach(),
                **self._profile_episode_tensors(prefix="step_profile"),
            },
        }
        return self.obs_buf, self.privileged_obs_buf, rewards, dones, self.extras

    def _terrain_episode_metrics(self, dones: torch.Tensor, reached_goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not hasattr(self.env, "env_terrain_map"):
            return {}
        terrain_ids = self.env.env_terrain_map.to(device=self.device, dtype=torch.long)
        if terrain_ids.numel() != self.num_envs:
            return {}
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            return {}

        metrics: Dict[str, torch.Tensor] = {}
        done_terrain_ids = terrain_ids[done_ids]
        max_terrain_id = int(terrain_ids.max().item()) if terrain_ids.numel() > 0 else -1
        for terrain_id in range(max_terrain_id + 1):
            mask = done_terrain_ids == int(terrain_id)
            if not torch.any(mask):
                continue
            terrain_done_ids = done_ids[mask]
            prefix = self._terrain_metric_prefix(terrain_id)
            metrics[f"{prefix}_success_rate"] = reached_goal[terrain_done_ids].float()
            metrics[f"{prefix}_episode_count"] = torch.ones(
                terrain_done_ids.numel(),
                dtype=torch.float32,
                device=self.device,
            )
        return metrics

    def _terrain_metric_prefix(self, terrain_id: int) -> str:
        name = self._terrain_name(terrain_id)
        if name is None:
            return f"terrain_{int(terrain_id)}"
        safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
        if safe_name == "":
            return f"terrain_{int(terrain_id)}"
        return f"terrain_{int(terrain_id)}_{safe_name}"

    def _terrain_name(self, terrain_id: int) -> Optional[str]:
        terrain_loader = getattr(self.env, "terrain_loader", None)
        if terrain_loader is None:
            return None
        instances = getattr(terrain_loader, "terrain_instances", None)
        if instances is None:
            instances = getattr(terrain_loader, "_instances", None)
        if instances is None or int(terrain_id) < 0 or int(terrain_id) >= len(instances):
            return None
        return getattr(instances[int(terrain_id)], "name", None)

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    @torch.no_grad()
    def _refresh_observation(self) -> None:
        if self.observation_mode == "map":
            self.obs_buf = self.obs_builder.build_observation(
                self.env,
                self.neural_map,
                self.goal_xy,
                previous_command=self.current_command,
            )
        elif self.observation_mode == "condition_only":
            self.obs_buf = self.obs_builder.build_condition_observation(
                self.env,
                self.goal_xy,
                previous_command=self.current_command,
            )
        else:
            raise ValueError("rl_nav.observation_mode must be 'map' or 'condition_only'")
        self.previous_distance_to_goal = torch.norm(
            self.goal_xy - self.env.root_local_pos[:, :2].to(self.device),
            dim=-1,
        )
        if bool(self.cfg.debug_observation_nan):
            self._debug_check_observation_finite()

    def _debug_check_observation_finite(self) -> None:
        finite = torch.isfinite(self.obs_buf)
        if bool(torch.all(finite)):
            return

        bad_rows = (~finite).any(dim=1).nonzero(as_tuple=False).flatten()
        bad_cols = (~finite[bad_rows]).nonzero(as_tuple=False)
        sample_envs = bad_rows[:8]
        sample_cols = bad_cols[:16]
        print("[RLNav][debug_observation_nan] non-finite obs_buf detected after _refresh_observation()")
        print(f"  bad_env_count={int(bad_rows.numel())} obs_shape={tuple(self.obs_buf.shape)}")
        print(f"  sample_bad_env_ids={sample_envs.detach().cpu().tolist()}")
        print(f"  sample_bad_obs_indices={sample_cols.detach().cpu().tolist()}")
        self._debug_print_tensor_stats("obs_buf", self.obs_buf)
        self._debug_print_tensor_stats("goal_xy", self.goal_xy)
        if hasattr(self.env, "root_states"):
            self._debug_print_tensor_stats("env.root_states", self.env.root_states)
        if hasattr(self.env, "root_local_pos"):
            self._debug_print_tensor_stats("env.root_local_pos", self.env.root_local_pos)
        if hasattr(self.env, "base_lin_vel"):
            self._debug_print_tensor_stats("env.base_lin_vel", self.env.base_lin_vel)
        if hasattr(self.env, "base_ang_vel"):
            self._debug_print_tensor_stats("env.base_ang_vel", self.env.base_ang_vel)
        if hasattr(self.neural_map, "maps"):
            self._debug_print_tensor_stats("neural_map.maps", self.neural_map.maps)
        if hasattr(self.neural_map, "uncertainties"):
            self._debug_print_tensor_stats("neural_map.uncertainties", self.neural_map.uncertainties)
        if hasattr(self.env, "env_terrain_map"):
            terrain_ids = self.env.env_terrain_map[sample_envs.to(device=self.env.env_terrain_map.device)]
            print(f"  sample_terrain_ids={terrain_ids.detach().cpu().tolist()}")
        raise RuntimeError("Non-finite RL navigation observation detected. See debug_observation_nan output above.")

    @staticmethod
    def _debug_print_tensor_stats(name: str, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            return
        detached = value.detach()
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        total_count = int(detached.numel())
        flat = detached.reshape(-1)
        flat_finite = finite.reshape(-1)
        sample_limit = 1_000_000
        if total_count > sample_limit:
            stride = max(1, total_count // sample_limit)
            flat = flat[::stride]
            flat_finite = flat_finite[::stride]
        if bool(torch.any(flat_finite)):
            finite_values = flat[flat_finite]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            mean_value = float(finite_values.float().mean().item())
            stats = f"min={min_value:.6g} max={max_value:.6g} mean={mean_value:.6g}"
        else:
            stats = "min=nan max=nan mean=nan"
        print(
            f"  {name}: shape={tuple(detached.shape)} "
            f"finite={finite_count}/{total_count} {stats}"
        )

    def _commands_to_env(self, commands: torch.Tensor) -> torch.Tensor:
        env_cmd = torch.zeros_like(self.env.commands)
        env_cmd[:, :3] = commands.to(device=self.env.device, dtype=self.env.commands.dtype)
        return env_cmd

    def _apply_command_to_locomotion_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[1] < 3 or not hasattr(self.env, "commands_scale"):
            return obs
        obs = obs.clone()
        obs[:, :3] = self.env.commands[:, :3] * self.env.commands_scale
        return obs

    def _compute_reward(
        self,
        *,
        command: torch.Tensor,
        previous_command: torch.Tensor,
        previous_distance: torch.Tensor,
        current_distance: torch.Tensor,
        reached_goal: torch.Tensor,
        collision: torch.Tensor,
        fixed_collision: torch.Tensor,
        collision_penalty_scale: torch.Tensor,
        repeated_collision_penalty: torch.Tensor,
        terminated: torch.Tensor,
        reached_middle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if self.cfg.reward.variant == "shaped":
            progress = previous_distance - current_distance
            if self.cfg.reward.progress_positive_only:
                progress = torch.clamp(progress, min=0.0)
            reward += float(self.cfg.reward.progress_scale) * progress
        elif self.cfg.reward.variant != "sparse":
            raise ValueError(f"Unknown reward variant: {self.cfg.reward.variant}")

        reward += reached_goal.float() * float(self.cfg.reward.reached_goal_reward)
        if reached_middle is not None:
            reward += reached_middle.float() * float(self.cfg.reward.middle_point_reward)
        reward -= collision.float() * float(self.cfg.reward.collision_penalty) * collision_penalty_scale
        reward -= repeated_collision_penalty
        reward -= fixed_collision.float() * float(self.cfg.reward.collision_penalty)
        reward -= terminated.float() * float(self.cfg.reward.termination_penalty)
        reward -= float(self.cfg.reward.time_penalty)
        reward -= command.square().mean(dim=-1) * float(self.cfg.reward.command_penalty_scale)
        reward -= (command - previous_command).square().mean(dim=-1) * float(
            self.cfg.reward.command_smoothness_penalty_scale
        )
        return reward

    def _collision_penalty_scale(self) -> torch.Tensor:
        cfg = self.cfg.reward.collision_uncertainty_scale
        if not bool(cfg.enabled):
            return torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        if str(cfg.scope) != "crop_mean":
            raise ValueError("reward.collision_uncertainty_scale.scope must be 'crop_mean'")
        min_scale = float(cfg.min_scale)
        max_scale = float(cfg.max_scale)
        if min_scale < 0.0 or max_scale < min_scale:
            raise ValueError("collision uncertainty scale requires 0 <= min_scale <= max_scale")
        confidence = self.obs_builder.crop_uncertainty_confidence(self.env, self.neural_map)
        confidence = torch.clamp(confidence.to(device=self.device, dtype=torch.float32), min=0.0, max=1.0)
        return min_scale + confidence * (max_scale - min_scale)

    def _new_repeated_collision_map(self) -> torch.Tensor:
        width, height = self._collision_map_shape()
        return torch.zeros((self.num_envs, height, width), dtype=torch.float32, device=self.device)

    def _collision_map_shape(self) -> tuple[int, int]:
        resolution = float(self.neural_map.cfg.least_count)
        if resolution <= 0.0:
            raise ValueError("neural_map.cfg.least_count must be positive")
        width = max(1, int(torch.ceil(torch.tensor(
            (float(self.neural_map.cfg.x_range[1]) - float(self.neural_map.cfg.x_range[0])) / resolution
        )).item()))
        height = max(1, int(torch.ceil(torch.tensor(
            (float(self.neural_map.cfg.y_range[1]) - float(self.neural_map.cfg.y_range[0])) / resolution
        )).item()))
        return width, height

    def _repeated_collision_penalty(self, collision: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.reward.repeated_collision_penalty
        penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not bool(cfg.enabled):
            return penalty

        radius_cells = int(cfg.radius_cells)
        max_count = float(cfg.max_count)
        decay = float(cfg.decay)
        if radius_cells < 0:
            raise ValueError("reward.repeated_collision_penalty.radius_cells must be non-negative")
        if max_count <= 0.0:
            raise ValueError("reward.repeated_collision_penalty.max_count must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("reward.repeated_collision_penalty.decay must be in (0, 1]")

        if decay < 1.0:
            self.repeated_collision_map.mul_(decay)

        collision = collision.to(device=self.device, dtype=torch.bool).view(-1)
        collision_ids = collision.nonzero(as_tuple=False).flatten()
        if collision_ids.numel() == 0:
            return penalty

        xy = self.env.root_local_pos[:, :2].to(device=self.device, dtype=torch.float32)
        x_idx, y_idx, valid = self._collision_xy_to_indices(xy)
        for env_id in collision_ids.tolist():
            if not bool(valid[env_id]):
                continue
            x = int(x_idx[env_id].item())
            y = int(y_idx[env_id].item())
            x0 = max(0, x - radius_cells)
            x1 = min(self.repeated_collision_map.shape[2], x + radius_cells + 1)
            y0 = max(0, y - radius_cells)
            y1 = min(self.repeated_collision_map.shape[1], y + radius_cells + 1)
            prior_count = self.repeated_collision_map[env_id, y0:y1, x0:x1].sum()
            normalized_count = torch.clamp(prior_count, min=0.0, max=max_count) / max_count
            penalty[env_id] = normalized_count * float(cfg.penalty_scale)
            self.repeated_collision_map[env_id, y, x] += 1.0
        return penalty

    def _collision_xy_to_indices(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = xy.to(device=self.device, dtype=torch.float32)
        resolution = float(self.neural_map.cfg.least_count)
        x_min = float(self.neural_map.cfg.x_range[0])
        y_min = float(self.neural_map.cfg.y_range[0])
        x_idx = torch.floor((xy[:, 0] - x_min) / resolution).to(dtype=torch.long)
        y_idx = torch.floor((xy[:, 1] - y_min) / resolution).to(dtype=torch.long)
        valid = (
            torch.isfinite(xy[:, 0])
            & torch.isfinite(xy[:, 1])
            & (x_idx >= 0)
            & (x_idx < self.repeated_collision_map.shape[2])
            & (y_idx >= 0)
            & (y_idx < self.repeated_collision_map.shape[1])
        )
        return x_idx, y_idx, valid

    def _handle_done_resets(
        self,
        *,
        reached_goal: torch.Tensor,
        low_level_done: torch.Tensor,
        timeout: torch.Tensor,
        collision_done: torch.Tensor,
    ) -> None:
        reached_ids = reached_goal.nonzero(as_tuple=False).flatten()
        if reached_ids.numel() > 0:
            self.success_count_since_map_reset[reached_ids] += 1

        map_reset_mask = reached_goal & (
            self.success_count_since_map_reset >= int(self.cfg.map_reset_success_threshold)
        )
        map_reset_ids = map_reset_mask.nonzero(as_tuple=False).flatten()
        goal_task_ids = (reached_goal & ~map_reset_mask).nonzero(as_tuple=False).flatten()
        if goal_task_ids.numel() > 0:
            self._reset_task(
                goal_task_ids,
                clear_map=False,
                resample_terrain=bool(self.cfg.terrain_resampling.on_task_success),
            )
        if map_reset_ids.numel() > 0:
            self._reset_map_episode(map_reset_ids)

        handled = reached_goal
        low_level_only_ids = (low_level_done & ~handled).nonzero(as_tuple=False).flatten()
        if low_level_only_ids.numel() > 0:
            self._reset_robot_to_task_start(low_level_only_ids)
            self._reset_episode_buffers(low_level_only_ids)

        timeout_ids = (timeout & ~handled & ~low_level_done).nonzero(as_tuple=False).flatten()
        if timeout_ids.numel() > 0:
            if self.cfg.resample_task_on_timeout:
                self._reset_task(
                    timeout_ids,
                    clear_map=False,
                    resample_terrain=bool(self.cfg.terrain_resampling.on_timeout),
                )
            else:
                self._reset_robot_to_task_start(timeout_ids)
                self._reset_episode_buffers(timeout_ids)

        collision_ids = (collision_done & ~handled & ~low_level_done & ~timeout).nonzero(as_tuple=False).flatten()
        if collision_ids.numel() > 0:
            if self.cfg.resample_task_on_collision_done:
                self._reset_task(
                    collision_ids,
                    clear_map=False,
                    resample_terrain=bool(self.cfg.terrain_resampling.on_collision_done),
                )
            else:
                self._reset_robot_to_task_start(collision_ids)
                self._reset_episode_buffers(collision_ids)

    def _reset_map_episode(self, env_ids: torch.Tensor) -> None:
        self._reset_task(
            env_ids,
            clear_map=True,
            resample_terrain=bool(self.cfg.terrain_resampling.on_map_reset),
        )
        self.success_count_since_map_reset[env_ids] = 0
        self.repeated_collision_map[env_ids] = 0.0

    def _reset_task(self, env_ids: torch.Tensor, *, clear_map: bool, resample_terrain: bool = False) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_env = env_ids.to(device=self.env.device)
        if bool(resample_terrain) and hasattr(self.env, "terrain_reset"):
            result = self.env.terrain_reset(env_ids_env, resample_goal=False)
            if result.start_xy.numel() > 0:
                self.task_start_xy[env_ids] = result.start_xy.to(device=self.device, dtype=torch.float32)
        elif hasattr(self.env, "task_reset"):
            result = self.env.task_reset(env_ids_env, resample_goal=False)
            if result.start_xy.numel() > 0:
                self.task_start_xy[env_ids] = result.start_xy.to(device=self.device, dtype=torch.float32)
        else:
            self.env.reset_idx(env_ids_env)
        self._refresh_root_local_pos(env_ids)
        if hasattr(self.env, "compute_observations"):
            self.env.compute_observations()
        self._store_current_start(env_ids)
        if clear_map:
            self._reset_maps(env_ids)
        self._reset_episode_buffers(env_ids)
        self.current_command[env_ids] = 0.0
        self.previous_command[env_ids] = 0.0
        self._sample_goals(env_ids)
        self._sample_middle_points(env_ids)

    def _reset_episode_buffers(self, env_ids: torch.Tensor) -> None:
        self.episode_length_buf[env_ids] = 0
        self.collision_count[env_ids] = 0

    def _sample_goals(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        cfg = self.cfg.goal_sampling
        if self._can_sample_goals_from_terrain_loader() and (
            float(cfg.min_distance) > 0.0
            or cfg.max_distance is not None
            or bool(cfg.avoid_previous)
        ):
            sampled = self._sample_goals_with_distance_filter(env_ids)
            self.goal_xy[env_ids] = sampled.to(device=self.device, dtype=torch.float32)
            self._mark_goals_initialized(env_ids)
            self._sync_env_task_goals(env_ids, sampled)
            return

        env_ids_env = env_ids.to(device=self.env.device)
        if hasattr(self.env, "resample_goals"):
            result = self.env.resample_goals(env_ids_env)
            assert result.goal_xy is not None
            sampled = result.goal_xy
        else:
            sampled = self.env.sample_goals(env_ids_env)[:, :2]
        self.goal_xy[env_ids] = sampled.to(device=self.device, dtype=torch.float32)
        self._mark_goals_initialized(env_ids)
        self._sync_env_task_goals(env_ids, self.goal_xy[env_ids])

    def _can_sample_goals_from_terrain_loader(self) -> bool:
        return hasattr(self.env, "terrain_loader") and hasattr(self.env, "env_terrain_map")

    def _sample_goals_with_distance_filter(self, env_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.goal_sampling
        if int(cfg.max_attempts) <= 0:
            raise ValueError("goal_sampling.max_attempts must be positive")
        if str(cfg.fallback) not in ("accept_last", "keep_previous"):
            raise ValueError("goal_sampling.fallback must be 'accept_last' or 'keep_previous'")
        if float(cfg.previous_tolerance) < 0.0:
            raise ValueError("goal_sampling.previous_tolerance must be non-negative")

        env_ids_env = env_ids.to(device=self.env.device, dtype=torch.long)
        reference_xy = self.env.root_local_pos[env_ids_env, :2].to(device=self.device, dtype=torch.float32)
        previous_goal_xy = self.goal_xy[env_ids].to(device=self.device, dtype=torch.float32)
        goal_initialized = self._goal_initialized(env_ids)
        remaining = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        selected = self.goal_xy[env_ids].clone()
        last_sampled = selected.clone()

        for _ in range(int(cfg.max_attempts)):
            remaining_ids = remaining.nonzero(as_tuple=False).flatten()
            if remaining_ids.numel() == 0:
                break
            candidate_env_ids = env_ids_env[remaining_ids.to(device=self.env.device)]
            terrain_ids = self.env.env_terrain_map[candidate_env_ids]
            sampled = self.env.terrain_loader.sample_goals(terrain_ids).squeeze(dim=1)[:, :2]
            sampled = sampled.to(device=self.device, dtype=torch.float32)
            last_sampled[remaining_ids] = sampled
            distance = torch.norm(sampled - reference_xy[remaining_ids], dim=-1)
            valid = distance >= float(cfg.min_distance)
            if cfg.max_distance is not None:
                valid &= distance <= float(cfg.max_distance)
            if bool(cfg.avoid_previous):
                previous_distance = torch.norm(sampled - previous_goal_xy[remaining_ids], dim=-1)
                valid &= (~goal_initialized[remaining_ids]) | (previous_distance > float(cfg.previous_tolerance))
            if torch.any(valid):
                valid_ids = remaining_ids[valid]
                selected[valid_ids] = sampled[valid]
                remaining[valid_ids] = False

        if torch.any(remaining):
            if str(cfg.fallback) == "accept_last":
                selected[remaining] = last_sampled[remaining]
            elif str(cfg.fallback) == "keep_previous":
                selected[remaining] = self.goal_xy[env_ids][remaining]
        return selected

    def _goal_initialized(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "goal_initialized"):
            return torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        return self.goal_initialized[env_ids.to(device=self.device, dtype=torch.long)]

    def _mark_goals_initialized(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self, "goal_initialized"):
            self.goal_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_initialized[env_ids.to(device=self.device, dtype=torch.long)] = True

    def _sync_env_task_goals(self, env_ids: torch.Tensor, goal_xy: torch.Tensor) -> None:
        if hasattr(self.env, "task_goal_xy"):
            env_ids_env = env_ids.to(device=self.env.device, dtype=torch.long)
            self.env.task_goal_xy[env_ids_env] = goal_xy.to(device=self.env.device, dtype=self.env.task_goal_xy.dtype)
            if hasattr(self.env, "task_goal_valid"):
                self.env.task_goal_valid[env_ids_env] = True

    def _middle_point_status(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_middle_point_buffers()
        middle_valid = self.middle_valid.clone()
        middle_distance = torch.norm(
            self.middle_xy - self.env.root_local_pos[:, :2].to(self.device),
            dim=-1,
        )
        reached_middle = (
            middle_valid
            & ~self.middle_reward_claimed
            & (middle_distance <= float(self.cfg.reward.middle_point_tolerance))
        )
        return middle_valid, middle_distance, reached_middle

    def _sample_middle_points(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._ensure_middle_point_buffers()
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.middle_reward_claimed[env_ids] = False
        self.middle_valid[env_ids] = False
        self.middle_xy[env_ids] = 0.0
        if not hasattr(self.env, "terrain_loader") or not hasattr(self.env, "env_terrain_map"):
            return
        terrain_loader = self.env.terrain_loader
        if not hasattr(terrain_loader, "get_num_middle_points") or not hasattr(terrain_loader, "sample_middle_points"):
            return

        env_ids_env = env_ids.to(device=self.env.device, dtype=torch.long)
        terrain_ids = self.env.env_terrain_map[env_ids_env]
        num_middle_points = terrain_loader.get_num_middle_points()[terrain_ids].to(device=self.device)
        valid = num_middle_points > 0
        if torch.any(valid):
            valid_ids = valid.nonzero(as_tuple=False).flatten()
            valid_env_ids = env_ids_env[valid_ids.to(device=self.env.device)]
            valid_terrain_ids = self.env.env_terrain_map[valid_env_ids]
            sampled = terrain_loader.sample_middle_points(valid_terrain_ids).squeeze(dim=1)[:, :2]
            self.middle_xy[env_ids[valid_ids]] = sampled.to(device=self.device, dtype=torch.float32)
            self.middle_valid[env_ids[valid_ids]] = True

    def _ensure_middle_point_buffers(self) -> None:
        if hasattr(self, "middle_xy") and hasattr(self, "middle_valid") and hasattr(self, "middle_reward_claimed"):
            return
        self.middle_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.middle_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.middle_reward_claimed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _freeze_models(self) -> None:
        self.locomotion_model.eval()
        for param in self.locomotion_model.parameters():
            param.requires_grad_(False)

    def _update_maps_preserving_done_envs(
        self,
        *,
        done_mask: torch.Tensor,
        long_latents: torch.Tensor,
        short_latents: torch.Tensor,
        xy_coord: torch.Tensor,
        root_velocity: torch.Tensor,
    ) -> None:
        done_mask = done_mask.to(device=self.device).view(-1).bool()
        saved_neural = self._snapshot_map_envs(self.neural_map, done_mask)
        self.neural_map.update(
            long_latents=self._replace_masked_rows(long_latents, done_mask),
            short_latents=self._replace_masked_rows(short_latents, done_mask),
            xy_coord=self._replace_masked_xy(xy_coord, done_mask),
            root_velocity=self._replace_masked_rows(root_velocity, done_mask),
        )
        self._restore_map_envs(self.neural_map, done_mask, saved_neural)

    def _snapshot_map_envs(self, map_obj, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not torch.any(mask):
            return {}
        obj_mask = mask.to(device=map_obj.device)
        snapshot = {}
        for name in ("maps", "long_maps", "short_maps", "uncertainties", "last_update_steps"):
            value = getattr(map_obj, name, None)
            if isinstance(value, torch.Tensor) and value.shape[0] == self.num_envs:
                snapshot[name] = value[obj_mask].clone()
        return snapshot

    def _restore_map_envs(self, map_obj, mask: torch.Tensor, snapshot: Dict[str, torch.Tensor]) -> None:
        if not snapshot:
            return
        obj_mask = mask.to(device=map_obj.device)
        for name, value in snapshot.items():
            target = getattr(map_obj, name, None)
            if isinstance(target, torch.Tensor):
                target[obj_mask] = value

    def _reset_maps(self, env_ids: torch.Tensor) -> None:
        reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reset_mask[env_ids] = True
        try:
            self.neural_map.reset(reset_mask, None)
        except TypeError:
            self.neural_map.reset(reset_mask)
        map_mask = reset_mask.to(device=self.neural_map.device).view(-1) != 0
        if hasattr(self.neural_map, "maps"):
            self.neural_map.maps[map_mask] = 0.0
        if hasattr(self.neural_map, "uncertainties"):
            initial_uncertainty = float(getattr(self.neural_map.cfg, "initial_uncertainty", 0.0))
            self.neural_map.uncertainties[map_mask] = initial_uncertainty
        if hasattr(self.neural_map, "last_update_steps"):
            current_step = getattr(self.neural_map, "current_step", 0)
            self.neural_map.last_update_steps[map_mask] = current_step

    def _refresh_root_local_pos(self, env_ids: torch.Tensor) -> None:
        if not hasattr(self.env, "terrain_loader") or not hasattr(self.env, "env_terrain_map"):
            return
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        terrain_origins = self.env.terrain_loader.get_terrain_origin(self.env.env_terrain_map[env_ids])
        self.env.root_local_pos[env_ids] = self.env.root_states[env_ids, :3] - terrain_origins

    def _store_current_start(self, env_ids: torch.Tensor) -> None:
        self._refresh_root_local_pos(env_ids)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.task_start_xy[env_ids] = self.env.root_local_pos[env_ids.to(device=self.env.device), :2].to(self.device)
        if hasattr(self.env, "task_start_xy") and hasattr(self.env, "task_start_valid"):
            env_ids_env = env_ids.to(device=self.env.device)
            self.env.task_start_xy[env_ids_env] = self.task_start_xy[env_ids].to(device=self.env.device)
            self.env.task_start_valid[env_ids_env] = True

    def _reset_robot_to_task_start(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_env = env_ids.to(device=self.env.device, dtype=torch.long)
        if hasattr(self.env, "robot_reset"):
            if hasattr(self.env, "task_start_xy") and hasattr(self.env, "task_start_valid"):
                self.env.task_start_xy[env_ids_env] = self.task_start_xy[env_ids.to(device=self.device)].to(self.env.device)
                self.env.task_start_valid[env_ids_env] = True
            self.env.robot_reset(env_ids_env)
            self._refresh_root_local_pos(env_ids)
            if hasattr(self.env, "compute_observations"):
                self.env.compute_observations()
            return
        terrain_ids = self.env.env_terrain_map[env_ids_env]
        start_xy = self.task_start_xy[env_ids.to(device=self.device)].to(device=self.env.device)
        sampled = torch.zeros((env_ids.numel(), 3), dtype=self.env.root_states.dtype, device=self.env.device)
        sampled[:, :2] = start_xy
        sampled[:, 2] = self.env.terrain_loader.get_height(terrain_ids, sampled[:, :2])
        world_sampled = self.env.terrain_loader.local_to_world(terrain_ids, sampled)
        self.env.env_origins[env_ids_env, :3] = world_sampled
        self.env.env_origins[env_ids_env, 2] = sampled[:, 2]
        self.env._reset_dofs(env_ids_env)
        self.env._reset_root_states(env_ids_env)
        self.env.actions[env_ids_env] = 0.0
        self.env.last_actions[env_ids_env] = 0.0
        if hasattr(self.env, "last_last_actions"):
            self.env.last_last_actions[env_ids_env] = 0.0
        self.env.last_dof_vel[env_ids_env] = 0.0
        if hasattr(self.env, "feet_air_time"):
            self.env.feet_air_time[env_ids_env] = 0.0
        if hasattr(self.env, "obs_history_buf"):
            self.env.obs_history_buf[env_ids_env] = 0.0
        if hasattr(self.env, "short_obs_history_buf"):
            self.env.short_obs_history_buf[env_ids_env] = 0.0
        self.env.episode_length_buf[env_ids_env] = 0
        self.env.reset_buf[env_ids_env] = 1
        self._refresh_root_local_pos(env_ids)
        if hasattr(self.env, "compute_observations"):
            self.env.compute_observations()

    def _collision_flags(self) -> torch.Tensor:
        if hasattr(self.env, "get_directional_collision_flags"):
            return self.env.get_directional_collision_flags().any(dim=-1).to(device=self.device, dtype=torch.bool)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _map_out_of_bounds_mask(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=self.device, dtype=torch.float32)
        x = xy[:, 0]
        y = xy[:, 1]
        return (
            (~torch.isfinite(x))
            | (~torch.isfinite(y))
            | (x < float(self.neural_map.cfg.x_range[0]))
            | (x >= float(self.neural_map.cfg.x_range[1]))
            | (y < float(self.neural_map.cfg.y_range[0]))
            | (y >= float(self.neural_map.cfg.y_range[1]))
        )

    def _replace_masked_xy(self, xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xy = xy.to(device=self.neural_map.device, dtype=self.neural_map.dtype).clone()
        mask = mask.to(device=self.neural_map.device, dtype=torch.bool).view(-1)
        if torch.any(mask):
            fallback = torch.tensor(
                [
                    float(self.neural_map.cfg.x_range[0]) + 0.5 * float(self.neural_map.cfg.least_count),
                    float(self.neural_map.cfg.y_range[0]) + 0.5 * float(self.neural_map.cfg.least_count),
                ],
                dtype=self.neural_map.dtype,
                device=self.neural_map.device,
            )
            xy[mask, :2] = fallback
        return xy

    def _replace_masked_rows(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value.clone()
        mask = mask.to(device=value.device, dtype=torch.bool).view(-1)
        if torch.any(mask):
            value[mask] = 0.0
        return value

    def _configure_low_level_command_mode(self) -> None:
        commands_cfg = getattr(getattr(self.env, "cfg", None), "commands", None)
        if commands_cfg is None:
            return
        if getattr(commands_cfg, "heading_command", False):
            raise ValueError(
                "RLNavPolicyEnv uses actions [vx, vy, yaw_rate], but env.cfg.commands.heading_command is True. "
                "Set env.commands.heading_command=false before constructing the environment."
            )
        if bool(self.cfg.disable_low_level_command_resampling):
            commands_cfg.resampling_time = 1.0e9

    @staticmethod
    def _normalize_env_step_output(step_out):
        if len(step_out) == 7:
            return step_out
        if len(step_out) == 5:
            obs, privileged_obs, rews, dones, infos = step_out
            return obs, privileged_obs, rews, dones, infos, None, None
        raise ValueError(f"Unexpected env.step output length: {len(step_out)}")

    def _new_profile(self) -> Dict[str, float]:
        if not bool(self.cfg.profiling_enabled):
            return {}
        return {
            "command_s": 0.0,
            "low_level_obs_patch_s": 0.0,
            "locomotion_s": 0.0,
            "env_step_s": 0.0,
            "post_step_bookkeeping_s": 0.0,
            "map_update_s": 0.0,
            "collision_check_s": 0.0,
            "low_level_reset_s": 0.0,
            "done_reset_s": 0.0,
            "obs_refresh_s": 0.0,
            "step_total_s": 0.0,
        }

    def _profile_now(self) -> Optional[float]:
        if not bool(self.cfg.profiling_enabled):
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def _profile_add(self, key: str, start: Optional[float]) -> None:
        if start is None:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._profile[key] = self._profile.get(key, 0.0) + (time.perf_counter() - start)

    def _profile_episode_tensors(self, *, prefix: str = "profile") -> Dict[str, torch.Tensor]:
        if not bool(self.cfg.profiling_enabled):
            return {}
        values = {}
        denom = max(1, int(self.steps_per_command))
        low_level_keys = {
            "low_level_obs_patch_s",
            "locomotion_s",
            "env_step_s",
            "post_step_bookkeeping_s",
            "map_update_s",
            "collision_check_s",
        }
        for key, value in self._profile.items():
            tensor_value = torch.full((self.num_envs,), float(value), dtype=torch.float32, device=self.device)
            values[f"{prefix}_{key}"] = tensor_value
            if key in low_level_keys:
                values[f"{prefix}_{key}_per_low_step"] = tensor_value / float(denom)
        return values
