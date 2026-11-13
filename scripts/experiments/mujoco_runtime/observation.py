from __future__ import annotations

from dataclasses import dataclass

import torch

from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg


@dataclass
class Go2LocomotionObservationBuilder:
    cfg: GO2RoughCfg
    device: torch.device

    def __post_init__(self) -> None:
        self.num_envs = 1
        self.long_history_length = int(self.cfg.env.num_long_obs_history_length)
        self.short_history_length = int(self.cfg.env.num_short_obs_history_length)
        self.proprio_dim = int(self.cfg.env.num_proprio_obs)
        self.obs_history_stride = int(self.cfg.env.obs_history_stride)
        self.obs_scales = self.cfg.normalization.obs_scales
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
            dtype=torch.float32,
            device=self.device,
        )
        self.short_history = torch.zeros(
            (self.num_envs, self.short_history_length, self.proprio_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.long_history = torch.zeros(
            (self.num_envs, self.long_history_length, self.proprio_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.step_count = 0

    def reset(self) -> None:
        self.short_history.zero_()
        self.long_history.zero_()
        self.step_count = 0

    @torch.no_grad()
    def build(self, env) -> torch.Tensor:
        imu_obs = torch.cat(
            (
                env.projected_gravity,
                env.base_ang_vel * float(self.obs_scales.ang_vel),
            ),
            dim=-1,
        )
        joint_obs = torch.cat(
            (
                (env.dof_pos - env.default_dof_pos) * float(self.obs_scales.dof_pos),
                env.dof_vel * float(self.obs_scales.dof_vel),
            ),
            dim=-1,
        )
        current_state = torch.cat((imu_obs, joint_obs, env.actions), dim=-1)
        if current_state.shape[-1] != self.proprio_dim:
            raise ValueError(f"Expected proprio dim {self.proprio_dim}, got {current_state.shape[-1]}")

        command_obs = env.commands[:, :3] * self.commands_scale
        estimated_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        terrain_index = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        current_augmented_state = torch.cat((current_state, estimated_velocity, terrain_index), dim=-1)

        self.short_history = torch.cat((self.short_history[:, 1:], current_state.unsqueeze(1)), dim=1)
        if self.step_count % self.obs_history_stride == 0:
            self.long_history = torch.cat((self.long_history[:, 1:], current_state.unsqueeze(1)), dim=1)
        self.step_count += 1

        return torch.cat(
            (
                command_obs,
                current_augmented_state,
                self.short_history.reshape(self.num_envs, -1),
                self.long_history.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )

    @torch.no_grad()
    def build_privileged(self, env) -> torch.Tensor:
        imu_obs = torch.cat(
            (
                env.projected_gravity,
                env.base_ang_vel * float(self.obs_scales.ang_vel),
            ),
            dim=-1,
        )
        joint_obs = torch.cat(
            (
                (env.dof_pos - env.default_dof_pos) * float(self.obs_scales.dof_pos),
                env.dof_vel * float(self.obs_scales.dof_vel),
            ),
            dim=-1,
        )
        current_state = torch.cat((imu_obs, joint_obs, env.actions), dim=-1)
        command_obs = env.commands[:, :3] * self.commands_scale
        obs_ground_truth = torch.cat((command_obs, current_state), dim=-1)

        num_height_points = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
        heights = torch.zeros((self.num_envs, num_height_points), dtype=torch.float32, device=self.device)
        privileged_extras = torch.zeros((self.num_envs, 42), dtype=torch.float32, device=self.device)
        return torch.cat(
            (
                obs_ground_truth,
                env.base_lin_vel * float(self.obs_scales.lin_vel),
                heights,
                privileged_extras,
            ),
            dim=-1,
        )
