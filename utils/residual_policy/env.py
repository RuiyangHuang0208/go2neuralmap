from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional

import torch

from rsl_rl.env import VecEnv
from utils.diffusion_policy.dataset import training_target_to_commands

from .command_mixer import CommandMixerConfig, mix_residual_commands, wrap_pi
from .observation import ResidualObservationBuilder, ResidualObservationConfig


@dataclass
class ResidualRewardConfig:
    variant: str = "sparse"
    goal_tolerance: float = 0.35
    reached_goal_reward: float = 10.0
    collision_penalty: float = 5.0
    termination_penalty: float = 10.0
    residual_penalty_scale: float = 0.01
    residual_vel_penalty_scale: Optional[float] = None
    residual_heading_penalty_scale: Optional[float] = None
    anti_cancellation_penalty_scale: float = 0.0
    progress_scale: float = 1.0
    progress_positive_only: bool = False
    time_penalty: float = 0.0


@dataclass
class ResidualEnvConfig:
    observation: ResidualObservationConfig = field(default_factory=ResidualObservationConfig)
    command_mixer: CommandMixerConfig = field(default_factory=CommandMixerConfig)
    reward: ResidualRewardConfig = field(default_factory=ResidualRewardConfig)
    command_dt: float = 0.5
    main_loop_hz: float = 50.0
    max_episode_steps: int = 80
    collision_done_threshold: Optional[int] = None
    map_reset_success_threshold: int = 5
    resample_task_on_timeout: bool = True
    resample_task_on_collision_done: bool = True


class ResidualPolicyEnv(VecEnv):
    """High-level residual RL wrapper around GO2NM.

    One wrapper step executes the full residual command horizon while the frozen
    locomotion policy runs at the low-level control rate.
    """

    def __init__(
        self,
        *,
        env,
        neural_map,
        cost_map,
        locomotion_model,
        cost_estimator,
        diffusion_algorithm,
        cfg: ResidualEnvConfig,
        device: torch.device | str,
    ):
        self.env = env
        self.neural_map = neural_map
        self.cost_map = cost_map
        self.locomotion_model = locomotion_model
        self.cost_estimator = cost_estimator
        self.diffusion_algorithm = diffusion_algorithm
        self.cfg = cfg
        self.device = torch.device(device)

        self.num_envs = int(env.num_envs)
        self.num_actions = int(cfg.command_mixer.residual_horizon) * 3
        self.max_episode_length = int(cfg.max_episode_steps)
        self.num_privileged_obs = None
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.extras: Dict = {}

        self.steps_per_command = max(1, int(round(float(cfg.command_dt) * float(cfg.main_loop_hz))))

        self.obs_builder = ResidualObservationBuilder(cfg.observation, device=self.device)
        encoder_output_dim = int(self.diffusion_algorithm.model.cfg.encoder.output_dim)
        self.num_obs = self.obs_builder.observation_dim(self.neural_map, encoder_output_dim)
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
        self.privileged_obs_buf = None
        self.goal_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.task_start_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.previous_distance_to_goal = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.collision_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_count_since_map_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_base_commands = torch.zeros(
            (self.num_envs, int(cfg.command_mixer.residual_horizon), 3),
            dtype=torch.float32,
            device=self.device,
        )

        self._freeze_models()
        self._store_current_start(torch.arange(self.num_envs, device=self.device))
        self._sample_goals(torch.arange(self.num_envs, device=self.device))
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
        actions = actions.to(device=self.device, dtype=torch.float32)
        base_commands = self.current_base_commands
        final_commands, mixer_info = mix_residual_commands(
            base_commands,
            actions,
            self.cfg.command_mixer,
            return_info=True,
        )
        previous_distance = torch.norm(self.goal_xy - self.env.root_local_pos[:, :2].to(self.device), dim=-1)

        any_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        any_low_level_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        accumulated_cost = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        low_level_obs = self.env.get_observations()
        for command_idx in range(final_commands.shape[1]):
            env_cmd = self._commands_to_env(final_commands[:, command_idx])
            self.env.commands[:, :] = env_cmd
            low_level_obs = self._apply_command_to_locomotion_obs(low_level_obs)
            for _ in range(self.steps_per_command):
                self.locomotion_model.compute_latents(low_level_obs.detach())
                low_level_actions = self.locomotion_model.act()
                step_out = self.env.step(low_level_actions.detach())
                low_level_obs, privileged_obs, _, dones, _, _, _ = self._normalize_env_step_output(step_out)
                step_done = dones.to(device=self.device, dtype=torch.bool)

                long_latent = self.locomotion_model.out["latent_long"]
                short_latent = self.locomotion_model.out["short_proj"]
                latent = torch.cat([long_latent, short_latent], dim=-1)
                local_pos = self.env.root_local_pos
                self._update_maps_preserving_done_envs(
                    done_mask=step_done,
                    long_latents=long_latent.clone(),
                    short_latents=short_latent.clone(),
                    xy_coord=local_pos[:, :2],
                    latent=latent,
                    privileged_obs=privileged_obs,
                    root_velocity=self.env.root_states[:, 7:9],
                    command=self.env.commands[:, :2],
                    directional_collision=self.env.get_directional_collision_flags(),
                    base_quat=self.env.base_quat,
                )
                cost = self.cost_estimator.compute_cost(
                    latent=latent,
                    privileged_obs=privileged_obs,
                    neural_map=self.neural_map,
                )
                collision_flags = self.env.get_directional_collision_flags().any(dim=-1)
                any_collision |= collision_flags
                any_low_level_done |= step_done
                accumulated_cost += cost.to(device=self.device, dtype=torch.float32).view(-1)
                if torch.any(step_done):
                    self._reset_robot_to_task_start(step_done.nonzero(as_tuple=False).flatten())
                    low_level_obs = self.env.get_observations()

        current_distance = torch.norm(self.goal_xy - self.env.root_local_pos[:, :2].to(self.device), dim=-1)
        reached_goal = current_distance <= float(self.cfg.reward.goal_tolerance)
        self.collision_count += any_collision.long()
        collision_done = torch.zeros_like(reached_goal)
        if self.cfg.collision_done_threshold is not None:
            collision_done = self.collision_count >= int(self.cfg.collision_done_threshold)

        self.episode_length_buf += 1
        timeout = self.episode_length_buf >= int(self.cfg.max_episode_steps)
        dones = reached_goal | any_low_level_done | timeout | collision_done
        rewards = self._compute_reward(
            actions=actions,
            previous_distance=previous_distance,
            current_distance=current_distance,
            reached_goal=reached_goal,
            collision=any_collision,
            terminated=any_low_level_done | collision_done,
            mixer_info=mixer_info,
        )

        self.rew_buf = rewards
        self.reset_buf = dones
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self._handle_done_resets(
                reached_goal=reached_goal,
                low_level_done=any_low_level_done,
                timeout=timeout,
                collision_done=collision_done,
            )

        self._refresh_observation()
        self.extras = {
            "episode": {
                "success": reached_goal.float(),
                "collision": any_collision.float(),
                "distance_to_goal": current_distance.detach(),
                "residual_norm": actions.view(self.num_envs, -1).norm(dim=-1).detach(),
                "progress": (previous_distance - current_distance).detach(),
                "positive_progress": torch.clamp(previous_distance - current_distance, min=0.0).detach(),
                "base_speed": mixer_info["base_speed"].mean(dim=-1).detach(),
                "mixed_speed": mixer_info["mixed_speed"].mean(dim=-1).detach(),
                "speed_ratio": mixer_info["speed_ratio"].mean(dim=-1).detach(),
                "vel_residual_norm": torch.norm(mixer_info["residual_delta_v"], dim=-1).mean(dim=-1).detach(),
                "heading_residual_abs": mixer_info["heading_delta"].abs().mean(dim=-1).detach(),
                "anti_cancellation": mixer_info["anti_cancellation"].mean(dim=-1).detach(),
                "low_level_cost": accumulated_cost.detach() / float(final_commands.shape[1] * self.steps_per_command),
            }
        }
        return self.obs_buf, self.privileged_obs_buf, rewards, dones, self.extras

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    @torch.no_grad()
    def _refresh_observation(self) -> None:
        batch = self.obs_builder.build_batch(self.env, self.neural_map, self.goal_xy)
        self.current_base_commands = self._sample_base_commands(batch)
        encoder = self.diffusion_algorithm.model.encode_condition
        self.obs_buf = self.obs_builder.build_observation(
            self.env,
            self.neural_map,
            self.goal_xy,
            encoder,
            batch=batch,
        ).to(device=self.device, dtype=torch.float32)
        self.previous_distance_to_goal = torch.norm(
            self.goal_xy - self.env.root_local_pos[:, :2].to(self.device),
            dim=-1,
        )

    @torch.no_grad()
    def _sample_base_commands(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        horizon = int(self.cfg.observation.model_horizon)
        if hasattr(self.diffusion_algorithm, "infer"):
            target = self.diffusion_algorithm.infer(batch, horizon)
        elif hasattr(self.diffusion_algorithm, "sample"):
            target = self.diffusion_algorithm.sample(batch, horizon)
        else:
            target = self.diffusion_algorithm.model(batch)
        commands = training_target_to_commands(
            target,
            velocity_scale=float(self.cfg.observation.velocity_scale),
        )
        commands[..., 2] = torch.remainder(commands[..., 2] + math.pi, 2.0 * math.pi) - math.pi
        return commands[:, : int(self.cfg.command_mixer.residual_horizon)].detach().to(self.device)

    def _commands_to_env(self, commands: torch.Tensor) -> torch.Tensor:
        env_cmd = torch.zeros_like(self.env.commands)
        env_cmd[:, 0:2] = commands[:, 0:2].to(device=self.env.device, dtype=self.env.commands.dtype)
        env_cmd[:, 3] = commands[:, 2].to(device=self.env.device, dtype=self.env.commands.dtype)
        if env_cmd.shape[1] > 2 and hasattr(self.env, "yaw"):
            heading_error = wrap_pi(env_cmd[:, 3] - self.env.yaw)
            env_cmd[:, 2] = torch.clamp(0.5 * heading_error, min=-1.0, max=1.0)
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
        actions: torch.Tensor,
        previous_distance: torch.Tensor,
        current_distance: torch.Tensor,
        reached_goal: torch.Tensor,
        collision: torch.Tensor,
        terminated: torch.Tensor,
        mixer_info: Dict[str, torch.Tensor],
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
        reward -= collision.float() * float(self.cfg.reward.collision_penalty)
        reward -= terminated.float() * float(self.cfg.reward.termination_penalty)
        reward -= float(self.cfg.reward.time_penalty)

        if (
            self.cfg.reward.residual_vel_penalty_scale is None
            and self.cfg.reward.residual_heading_penalty_scale is None
            and float(self.cfg.reward.anti_cancellation_penalty_scale) == 0.0
        ):
            residual_norm_sq = actions.view(self.num_envs, -1).square().mean(dim=-1)
            reward -= residual_norm_sq * float(self.cfg.reward.residual_penalty_scale)
            return reward

        residual_delta_v = mixer_info["residual_delta_v"]
        heading_delta = mixer_info["heading_delta"]
        vel_norm_sq = residual_delta_v.square().sum(dim=-1).mean(dim=-1)
        heading_sq = heading_delta.square().mean(dim=-1)
        anti_cancellation = mixer_info["anti_cancellation"].mean(dim=-1)

        vel_penalty_scale = (
            self.cfg.reward.residual_penalty_scale
            if self.cfg.reward.residual_vel_penalty_scale is None
            else self.cfg.reward.residual_vel_penalty_scale
        )
        heading_penalty_scale = (
            self.cfg.reward.residual_penalty_scale
            if self.cfg.reward.residual_heading_penalty_scale is None
            else self.cfg.reward.residual_heading_penalty_scale
        )
        reward -= vel_norm_sq * float(vel_penalty_scale)
        reward -= heading_sq * float(heading_penalty_scale)
        reward -= anti_cancellation * float(self.cfg.reward.anti_cancellation_penalty_scale)
        return reward

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
            self._reset_task(goal_task_ids, clear_map=False)
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
                self._reset_task(timeout_ids, clear_map=False)
            else:
                self._reset_robot_to_task_start(timeout_ids)
                self._reset_episode_buffers(timeout_ids)

        collision_ids = (collision_done & ~handled & ~low_level_done & ~timeout).nonzero(as_tuple=False).flatten()
        if collision_ids.numel() > 0:
            if self.cfg.resample_task_on_collision_done:
                self._reset_task(collision_ids, clear_map=False)
            else:
                self._reset_robot_to_task_start(collision_ids)
                self._reset_episode_buffers(collision_ids)

    def _reset_map_episode(self, env_ids: torch.Tensor) -> None:
        self._reset_task(env_ids, clear_map=True)
        self.success_count_since_map_reset[env_ids] = 0

    def _reset_task(self, env_ids: torch.Tensor, *, clear_map: bool) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_env = env_ids.to(device=self.env.device)
        if hasattr(self.env, "task_reset"):
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
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[env_ids] = True
            self._reset_maps(reset_mask)
        self._reset_episode_buffers(env_ids)
        self._sample_goals(env_ids)

    def _reset_episode_buffers(self, env_ids: torch.Tensor) -> None:
        self.episode_length_buf[env_ids] = 0
        self.collision_count[env_ids] = 0

    def _sample_goals(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids_env = env_ids.to(device=self.env.device)
        if hasattr(self.env, "resample_goals"):
            result = self.env.resample_goals(env_ids_env)
            assert result.goal_xy is not None
            sampled = result.goal_xy
        else:
            sampled = self.env.sample_goals(env_ids_env)[:, :2]
        self.goal_xy[env_ids] = sampled.to(device=self.device, dtype=torch.float32)

    def _freeze_models(self) -> None:
        self.diffusion_algorithm.model.eval()
        for param in self.diffusion_algorithm.model.parameters():
            param.requires_grad_(False)
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
        latent: torch.Tensor,
        privileged_obs: torch.Tensor,
        root_velocity: torch.Tensor,
        command: torch.Tensor,
        directional_collision: torch.Tensor,
        base_quat: torch.Tensor,
    ) -> None:
        done_mask = done_mask.to(device=self.device).view(-1).bool()
        saved_neural = self._snapshot_map_envs(self.neural_map, done_mask)
        saved_cost = self._snapshot_map_envs(self.cost_map, done_mask)
        self.neural_map.update(
            long_latents=long_latents,
            short_latents=short_latents,
            xy_coord=xy_coord,
            root_velocity=root_velocity,
        )
        cost = self.cost_estimator.compute_cost(
            latent=latent,
            privileged_obs=privileged_obs,
            neural_map=self.neural_map,
        )
        self.cost_map.update(
            current_pos=xy_coord,
            new_values=cost.clone(),
            root_velocity=root_velocity,
            command=command,
            directional_collision=directional_collision,
            base_quat=base_quat,
            latent=latent,
        )
        self._restore_map_envs(self.neural_map, done_mask, saved_neural)
        self._restore_map_envs(self.cost_map, done_mask, saved_cost)

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

    def _reset_maps(self, reset_mask: torch.Tensor) -> None:
        try:
            self.neural_map.reset(reset_mask, None)
        except TypeError:
            self.neural_map.reset(reset_mask)
        self.cost_map.reset(reset_mask)

        map_mask = reset_mask.to(device=self.neural_map.device).view(-1) != 0
        if hasattr(self.neural_map, "maps"):
            self.neural_map.maps[map_mask] = 0.0
        if hasattr(self.neural_map, "uncertainties"):
            initial_uncertainty = float(getattr(self.neural_map.cfg, "initial_uncertainty", 0.0))
            self.neural_map.uncertainties[map_mask] = initial_uncertainty
        if hasattr(self.neural_map, "last_update_steps"):
            current_step = getattr(self.neural_map, "current_step", 0)
            self.neural_map.last_update_steps[map_mask] = current_step

        cost_mask = reset_mask.to(device=self.cost_map.device).view(-1) != 0
        if hasattr(self.cost_map, "maps"):
            self.cost_map.maps[cost_mask] = float(getattr(self.cost_map.cfg, "init_value", 0.0))

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
                self.env.task_start_xy[env_ids_env] = self.task_start_xy[env_ids.to(device=self.device)].to(device=self.env.device)
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

    @staticmethod
    def _normalize_env_step_output(step_out):
        if len(step_out) == 7:
            return step_out
        if len(step_out) == 5:
            obs, privileged_obs, rews, dones, infos = step_out
            return obs, privileged_obs, rews, dones, infos, None, None
        raise ValueError(f"Unexpected env.step output length: {len(step_out)}")
