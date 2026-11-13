from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch

from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg


ISAAC_JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


@dataclass
class MuJoCoGo2EnvConfig:
    xml_path: str = "legged_gym/resources/robots/go2/mujoco/scene.xml"
    device: str = "cuda"
    control_dt: float = 0.02
    render: bool = False
    realtime: bool = False
    render_fps: float = 30.0
    goal_xy: Sequence[float] = (2.0, 0.0)
    goal_xy_range: Sequence[float] = (0.5, 3.0, -1.5, 1.5)
    reset_pos: Sequence[float] = (0.0, 0.0, 0.445)
    reset_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0)
    # reset_quat_wxyz: Sequence[float] = (0.0, 0.0, 0.0, 1.0)
    # reset_quat_wxyz: Sequence[float] = (0.00872654, 0.0, 0.0, 0.99996192)
    
    # reset_quat_wxyz: Sequence[float] = (np.sqrt(1/2), 0.0, 0.0, np.sqrt(1/2))
    torque_limit: float = 45.43
    terrain_id: int = 0
    sequential_terrain_points: bool = True
    joint_order: list[str] = field(default_factory=lambda: list(ISAAC_JOINT_ORDER))


class MuJoCoGo2EnvAdapter:
    """Single-env tensor facade over a MuJoCo Go2 model.

    The public attributes intentionally mirror the small IsaacGym env surface
    used by existing command generators, observation builders, and map modules.
    """

    def __init__(
        self,
        cfg: MuJoCoGo2EnvConfig,
        go2_cfg: GO2RoughCfg | None = None,
        terrain_loader: Any | None = None,
    ):
        self.cfg = cfg
        self.go2_cfg = go2_cfg or GO2RoughCfg()
        self.terrain_loader = terrain_loader
        self.device = torch.device(cfg.device if torch.cuda.is_available() or not cfg.device.startswith("cuda") else "cpu")
        self.num_envs = 1
        self.num_actions = 12
        self.model = mujoco.MjModel.from_xml_path(str(Path(cfg.xml_path).expanduser()))
        self.data = mujoco.MjData(self.model)
        self.physics_dt = float(self.model.opt.timestep)
        self.dt = float(cfg.control_dt)
        self.physics_steps_per_control = max(1, int(round(self.dt / self.physics_dt)))
        self.viewer = None

        self.joint_order = list(cfg.joint_order)
        self.joint_qpos_addr = np.array([self.model.joint(name).qposadr[0] for name in self.joint_order], dtype=np.int32)
        self.joint_qvel_addr = np.array([self.model.joint(name).dofadr[0] for name in self.joint_order], dtype=np.int32)
        self.actuator_by_joint = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator(actuator_id).trnid[0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            self.actuator_by_joint[joint_name] = actuator_id
        self.actuator_addr = np.array([self.actuator_by_joint[name] for name in self.joint_order], dtype=np.int32)

        self.obs_scales = self.go2_cfg.normalization.obs_scales
        self.default_dof_pos = self._default_dof_pos().view(1, -1)
        self.p_gains = torch.full((1, self.num_actions), 28.0, dtype=torch.float32, device=self.device)
        self.d_gains = torch.full((1, self.num_actions), 0.7, dtype=torch.float32, device=self.device)
        self.torque_limits = torch.full(
            (1, self.num_actions),
            float(cfg.torque_limit),
            dtype=torch.float32,
            device=self.device,
        )
        self.commands = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((1, self.num_actions), dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros((1, self.num_actions), dtype=torch.float32, device=self.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.root_states = torch.zeros((1, 13), dtype=torch.float32, device=self.device)
        self.root_local_pos = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.last_reset_local_pos = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.base_quat = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
        self.base_lin_vel = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vel = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.projected_gravity = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        self.yaw = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.step_cost = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.task_goal_xy = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
        self.task_goal_valid = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.env_terrain_map = torch.full(
            (1,),
            int(cfg.terrain_id),
            dtype=torch.long,
            device=self.device,
        )
        self.common_step_counter = 0
        self.max_episode_length = 1000000
        self.reset_idx(torch.tensor([0], dtype=torch.long, device=self.device))

    def reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = self._reset_world_pos().detach().cpu().numpy()
        self.data.qpos[3:7] = np.asarray(self.cfg.reset_quat_wxyz, dtype=np.float64)
        self.data.qpos[self.joint_qpos_addr] = self.default_dof_pos[0].detach().cpu().numpy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.actions.zero_()
        self.last_actions.zero_()
        self.reset_buf.zero_()
        self.refresh_tensors()

    def task_reset(self, env_ids: torch.Tensor, resample_goal: bool = False):
        self.reset_idx(env_ids)
        if resample_goal:
            return self.resample_goals(env_ids)
        from utils.reset_events import ResetResult

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        goal_xy = self.task_goal_xy[env_ids] if env_ids.numel() > 0 else torch.empty((0, 2), device=self.device)
        return ResetResult(
            env_ids=env_ids,
            reset_type="mujoco_task_reset",
            terrain_ids=torch.zeros_like(env_ids),
            start_xy=self.root_local_pos[env_ids, :2] if env_ids.numel() > 0 else goal_xy,
            goal_xy=goal_xy,
            root_local_pos=self.root_local_pos[env_ids] if env_ids.numel() > 0 else torch.empty((0, 3), device=self.device),
            done_mask=None,
        )

    def sample_goals(self, env_ids: torch.Tensor) -> torch.Tensor:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        if self.terrain_loader is not None:
            terrain_ids = self.env_terrain_map.index_select(0, env_ids)
            goals = self.terrain_loader.sample_goals(
                terrain_ids,
                sequential=bool(self.cfg.sequential_terrain_points),
            ).squeeze(1)
            return goals.to(device=self.device, dtype=torch.float32)
        x_min, x_max, y_min, y_max = [float(v) for v in self.cfg.goal_xy_range]
        goals = torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=self.device)
        goals[:, 0] = torch.empty(env_ids.numel(), device=self.device).uniform_(x_min, x_max)
        goals[:, 1] = torch.empty(env_ids.numel(), device=self.device).uniform_(y_min, y_max)
        return goals

    def resample_goals(self, env_ids: torch.Tensor):
        from utils.reset_events import ResetResult

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        goals = self.sample_goals(env_ids)
        if env_ids.numel() > 0:
            self.task_goal_xy[env_ids] = goals[:, :2]
            self.task_goal_valid[env_ids] = True
        goal_xy = goals[:, :2] if goals.numel() > 0 else torch.empty((0, 2), dtype=torch.float32, device=self.device)
        return ResetResult(
            env_ids=env_ids,
            reset_type="mujoco_goal_resample",
            terrain_ids=torch.zeros_like(env_ids),
            start_xy=self.root_local_pos[env_ids, :2] if env_ids.numel() > 0 else goal_xy,
            goal_xy=goal_xy,
            root_local_pos=self.root_local_pos[env_ids] if env_ids.numel() > 0 else torch.empty((0, 3), device=self.device),
            done_mask=None,
        )

    def set_fixed_goal(self) -> None:
        self.task_goal_xy[0] = torch.as_tensor(self.cfg.goal_xy, dtype=torch.float32, device=self.device)
        self.task_goal_valid[0] = True

    def step(self, actions: torch.Tensor) -> None:
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.to(device=self.device, dtype=torch.float32))
        torques = self.compute_torques(self.actions)[0].detach().cpu().numpy()
        self.data.ctrl[self.actuator_addr] = torques
        for _ in range(self.physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)
        self.common_step_counter += 1
        self.step_cost.zero_()
        self.refresh_tensors()

    def get_directional_collision_flags(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 4), dtype=torch.bool, device=self.device)

    def compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        actions_scaled = actions * float(self.go2_cfg.control.action_scale)
        torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def refresh_tensors(self) -> None:
        qpos = self.data.qpos
        qvel = self.data.qvel
        root_pos = torch.as_tensor(qpos[0:3], dtype=torch.float32, device=self.device)
        root_quat_xyzw = torch.as_tensor(
            [qpos[4], qpos[5], qpos[6], qpos[3]],
            dtype=torch.float32,
            device=self.device,
        )
        root_lin_vel = torch.as_tensor(qvel[0:3], dtype=torch.float32, device=self.device)
        root_ang_vel_local = torch.as_tensor(qvel[3:6], dtype=torch.float32, device=self.device)
        root_ang_vel_world = quat_rotate(root_quat_xyzw.view(1, 4), root_ang_vel_local.view(1, 3))[0]
        self.dof_pos[0] = torch.as_tensor(qpos[self.joint_qpos_addr], dtype=torch.float32, device=self.device)
        self.dof_vel[0] = torch.as_tensor(qvel[self.joint_qvel_addr], dtype=torch.float32, device=self.device)
        self.root_states[0, 0:3] = root_pos
        self.root_states[0, 3:7] = root_quat_xyzw
        self.root_states[0, 7:10] = root_lin_vel
        self.root_states[0, 10:13] = root_ang_vel_world
        self.root_local_pos[0] = self._world_to_local(root_pos.view(1, 3))[0]
        self.base_quat[0] = root_quat_xyzw
        self.base_lin_vel[0] = quat_rotate_inverse(root_quat_xyzw.view(1, 4), root_lin_vel.view(1, 3))[0]
        self.base_ang_vel[0] = root_ang_vel_local
        gravity = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device)
        self.projected_gravity[0] = quat_rotate_inverse(root_quat_xyzw.view(1, 4), gravity)[0]
        self.yaw[0] = yaw_from_quat_xyzw(root_quat_xyzw)

    def _default_dof_pos(self) -> torch.Tensor:
        values = [float(self.go2_cfg.init_state.default_joint_angles[name]) for name in self.joint_order]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _reset_world_pos(self) -> torch.Tensor:
        if self.terrain_loader is None:
            reset_pos = torch.as_tensor(self.cfg.reset_pos, dtype=torch.float32, device=self.device).view(1, 3)
            self.last_reset_local_pos.copy_(reset_pos)
            return reset_pos[0]
        terrain_ids = self.env_terrain_map.to(device=self.device, dtype=torch.long)
        starts = self.terrain_loader.sample_starts(
            terrain_ids,
            sequential=bool(self.cfg.sequential_terrain_points),
        ).squeeze(1)
        starts = starts.to(device=self.device, dtype=torch.float32)
        base_height = float(self.cfg.reset_pos[2])
        starts[:, 2] = self.terrain_loader.get_height(terrain_ids, starts[:, :2]).to(
            device=self.device,
            dtype=torch.float32,
        ) + base_height
        self.last_reset_local_pos.copy_(starts)
        return self.terrain_loader.local_to_world(terrain_ids, starts)[0].to(device=self.device, dtype=torch.float32)

    def _world_to_local(self, world_pos: torch.Tensor) -> torch.Tensor:
        if self.terrain_loader is None:
            return world_pos.to(device=self.device, dtype=torch.float32)
        terrain_ids = self.env_terrain_map.to(device=self.device, dtype=torch.long)
        return self.terrain_loader.world_to_local(
            terrain_ids,
            world_pos.to(device=self.device, dtype=torch.float32),
        ).to(device=self.device, dtype=torch.float32)


def quat_rotate_inverse(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q_xyzw[:, :3]
    q_w = q_xyzw[:, 3:4]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v - 2.0 * (q_w * uv - uuv)


def quat_rotate(q_xyzw: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q_xyzw[:, :3]
    q_w = q_xyzw[:, 3:4]
    uv = torch.cross(q_xyz, v, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return v + 2.0 * (q_w * uv + uuv)


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)
