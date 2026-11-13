from isaacgym.torch_utils import to_torch, torch_rand_float, quat_rotate_inverse, quat_from_euler_xyz
from isaacgym import gymtorch, gymapi

import torch
import numpy as np

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, eu
from legged_gym.utils.math import euler_from_quaternion

try:
    from .go2_rough_config import GO2RoughCfg
except ImportError:
    from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg
    print("Import module from absolute path for debugging")

from typing import Union, Callable, Optional, Sequence
import os
from utils.reset_events import ResetResult
from utils.terrain_loader import TerrainLoader


class GO2Rough(LeggedRobot):
    def __init__(self, cfg: GO2RoughCfg, sim_params, physics_engine, sim_device, headless):

        self.long_history_length = cfg.env.num_long_obs_history_length
        self.short_history_length = cfg.env.num_short_obs_history_length
        self.dim_proprio_obs = cfg.env.num_proprio_obs

        self.sensor_map = {}
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._assign_terrain_to_env()
        self.reset()

    def _init_buffers(self):
        super()._init_buffers()
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)

        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.roll = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.pitch = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.foot_contact_sensor_force = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        # self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
        self.sensor_forces = force_sensor_readings.view(self.num_envs, -1, 6)[..., :3]
        self.foot_contact_sensor_force = self.sensor_forces[:, self.feet_sensor_idx]

        self.obs_history_buf = torch.zeros(
            size=(
                self.num_envs,
                self.long_history_length,
                self.dim_proprio_obs,
            ),
            device=self.device,
            dtype=torch.float
        )

        self.short_obs_history_buf = torch.zeros(
            size=(
                self.num_envs,
                self.short_history_length,
                self.dim_proprio_obs,
            ),
            device=self.device,
            dtype=torch.float
        )

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.last_last_actions = torch.zeros_like(self.last_actions)
        self.custom_origins = True

        self.env_terrain_map = torch.zeros((self.num_envs), dtype=torch.long, device=self.device)
        self.terrain_difficulties = self.terrain_loader.get_terrain_difficulties()

        self.root_local_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.task_start_xy = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.task_goal_xy = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self.task_start_valid = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)
        self.task_goal_valid = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)
        self.last_reset_result: Optional[ResetResult] = None

    def _assign_terrain_to_env(self, env_ids=None, terrain_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if terrain_ids is None:
            terrain_ids = torch.arange(len(self.terrain_difficulties), device=self.device, dtype=torch.long)
        # Uniformly sample id from terrain_ids and save it to self.env_terrain_map
        for i in env_ids:
            idx = torch.randint(0, len(terrain_ids), (), device=self.device)
            sampled_id = terrain_ids[idx]
            self.env_terrain_map[i] = sampled_id

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'trimesh':
            # self.terrain = Terrain(self.cfg.terrain, self.num_envs)
            self.cfg.terrain.device = self.device
            self.terrain_loader = TerrainLoader(self.cfg.terrain)
            self.terrain_loader.load_into_sim(self.sim, self.gym)
        self._create_envs()

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    def _create_envs(self, create_envs_cbk: Union[None, Callable] = None):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot

            Override LeggedRobot._create_envs(), in order to add force sensors
        """
        
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        self._add_force_sensors(robot_asset)

        if create_envs_cbk is not None:
            create_envs_cbk()

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def sample_terrain_for_env(self, terrain_ids, env_ids):
        pass

    def _add_force_sensors(self, robot_asset):

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.rigid_body_names = body_names
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        sensor_pose = gymapi.Transform()
        feet_sensor_idx = []

        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
            self.sensor_map[name] = sensor_idx
            feet_sensor_idx.append(sensor_idx)
        self.feet_sensor_idx = torch.tensor(feet_sensor_idx, device=self.device, dtype=torch.long)

    def sample_goals(self, env_ids):
        terrain_ids = self.env_terrain_map[env_ids]
        sampled = self.terrain_loader.sample_goals(terrain_ids).squeeze(dim=1) # Convert to (len_env_ids, 3)
        return sampled

    def sample_middle_points(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        terrain_ids = self.env_terrain_map[env_ids]
        return self.terrain_loader.sample_middle_points(terrain_ids).squeeze(dim=1)

    def has_middle_points(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        terrain_ids = self.env_terrain_map[env_ids]
        return self.terrain_loader.has_middle_points(terrain_ids)

    def resample_goals(
        self,
        env_ids,
        *,
        indices: Optional[torch.Tensor] = None,
        sequential: bool = False,
        reset_callbacks: Optional[Union[Callable[[ResetResult], None], Sequence[Callable[[ResetResult], None]]]] = None,
    ) -> ResetResult:
        env_ids = self._normalize_env_ids(env_ids)
        terrain_ids = self.env_terrain_map[env_ids]
        goals = self.terrain_loader.sample_goals(
            terrain_ids,
            indices=indices,
            sequential=sequential,
        ).squeeze(dim=1)
        self.task_goal_xy[env_ids] = goals[:, :2]
        self.task_goal_valid[env_ids] = True
        result = self._make_reset_result(env_ids, "goal", goal_xy=goals[:, :2])
        self._run_reset_callbacks(reset_callbacks, result)
        return result

    def robot_reset(
        self,
        env_ids,
        *,
        reset_callbacks: Optional[Union[Callable[[ResetResult], None], Sequence[Callable[[ResetResult], None]]]] = None,
    ) -> ResetResult:
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return self._make_reset_result(env_ids, "robot")

        missing_start = ~self.task_start_valid[env_ids]
        if torch.any(missing_start):
            missing_ids = env_ids[missing_start]
            terrain_ids = self.env_terrain_map[missing_ids]
            starts = self._sample_starts(terrain_ids)
            self._set_env_origins_from_starts(missing_ids, terrain_ids, starts)
            self.task_start_xy[missing_ids] = starts[:, :2]
            self.task_start_valid[missing_ids] = True

        terrain_ids = self.env_terrain_map[env_ids]
        starts = torch.zeros((env_ids.numel(), 3), dtype=self.root_states.dtype, device=self.device)
        starts[:, :2] = self.task_start_xy[env_ids]
        starts[:, 2] = self.terrain_loader.get_height(terrain_ids, starts[:, :2])
        self._set_env_origins_from_starts(env_ids, terrain_ids, starts)
        self._reset_robot_state_and_buffers(env_ids, log_episode=False)
        result = self._make_reset_result(env_ids, "robot")
        self.last_reset_result = result
        self._run_reset_callbacks(reset_callbacks, result)
        return result

    def task_reset(
        self,
        env_ids,
        *,
        resample_goal: bool = True,
        reset_callbacks: Optional[Union[Callable[[ResetResult], None], Sequence[Callable[[ResetResult], None]]]] = None,
    ) -> ResetResult:
        env_ids = self._normalize_env_ids(env_ids)
        terrain_ids = self.env_terrain_map[env_ids]
        starts = self._sample_starts(terrain_ids)
        self._set_env_origins_from_starts(env_ids, terrain_ids, starts)
        self.task_start_xy[env_ids] = starts[:, :2]
        self.task_start_valid[env_ids] = True
        goal_xy = None
        if resample_goal:
            goals = self.terrain_loader.sample_goals(terrain_ids).squeeze(dim=1)
            goal_xy = goals[:, :2]
            self.task_goal_xy[env_ids] = goal_xy
            self.task_goal_valid[env_ids] = True
        self._reset_robot_state_and_buffers(env_ids, log_episode=True)
        result = self._make_reset_result(env_ids, "task", goal_xy=goal_xy)
        self.last_reset_result = result
        self._run_reset_callbacks(reset_callbacks, result)
        return result

    def terrain_reset(
        self,
        env_ids,
        *,
        terrain_ids: Optional[torch.Tensor] = None,
        resample_goal: bool = True,
        reset_callbacks: Optional[Union[Callable[[ResetResult], None], Sequence[Callable[[ResetResult], None]]]] = None,
    ) -> ResetResult:
        env_ids = self._normalize_env_ids(env_ids)
        if terrain_ids is None:
            self._assign_terrain_to_env(env_ids)
        else:
            terrain_ids = terrain_ids.to(device=self.device, dtype=torch.long).view(-1)
            if terrain_ids.numel() == 1 and env_ids.numel() > 1:
                terrain_ids = terrain_ids.expand(env_ids.numel())
            if terrain_ids.numel() != env_ids.numel():
                raise ValueError(
                    f"terrain_ids must have one value or match env_ids length, got {terrain_ids.numel()} for {env_ids.numel()} envs"
                )
            self.env_terrain_map[env_ids] = terrain_ids

        terrain_ids = self.env_terrain_map[env_ids]
        starts = self._sample_starts(terrain_ids)
        self._set_env_origins_from_starts(env_ids, terrain_ids, starts)
        self.task_start_xy[env_ids] = starts[:, :2]
        self.task_start_valid[env_ids] = True
        goal_xy = None
        if resample_goal:
            goals = self.terrain_loader.sample_goals(terrain_ids).squeeze(dim=1)
            goal_xy = goals[:, :2]
            self.task_goal_xy[env_ids] = goal_xy
            self.task_goal_valid[env_ids] = True
        else:
            self.task_goal_valid[env_ids] = False
        self._reset_robot_state_and_buffers(env_ids, log_episode=True)
        result = self._make_reset_result(env_ids, "terrain", goal_xy=goal_xy)
        self.last_reset_result = result
        self._run_reset_callbacks(reset_callbacks, result)
        return result

    def reset_idx(self, env_ids):
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        # reset env origins
        if self.cfg.terrain.curriculum:
            # For example
            terrain_ids = torch.where(self.terrain_difficulties < 2)[0] # sample easier terrains for reset
            self._assign_terrain_to_env(env_ids, terrain_ids)
        terrain_ids = self.env_terrain_map[env_ids]
        sampled = self._sample_starts(terrain_ids)
        self._set_env_origins_from_starts(env_ids, terrain_ids, sampled)
        self.task_start_xy[env_ids] = sampled[:, :2]
        self.task_start_valid[env_ids] = True

        # reset robot states
        self._reset_robot_state_and_buffers(env_ids, log_episode=True)
        self.last_reset_result = self._make_reset_result(env_ids, "auto", done_mask=self.reset_buf.clone())

    def _normalize_env_ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, int):
            return torch.tensor([env_ids], dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

    def _sample_starts(self, terrain_ids: torch.Tensor) -> torch.Tensor:
        return self.terrain_loader.sample_starts(terrain_ids).squeeze(dim=1)

    def _set_env_origins_from_starts(self, env_ids: torch.Tensor, terrain_ids: torch.Tensor, starts: torch.Tensor) -> None:
        world_sampled = self.terrain_loader.local_to_world(terrain_ids, starts)
        self.env_origins[env_ids, :3] = world_sampled[:]
        self.env_origins[env_ids, 2] = self.terrain_loader.get_height(terrain_ids, starts[:, :2])

    def _reset_robot_state_and_buffers(self, env_ids: torch.Tensor, *, log_episode: bool) -> None:
        if env_ids.numel() == 0:
            return
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()

        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.

        self.obs_history_buf[env_ids] = 0.
        self.short_obs_history_buf[env_ids] = 0.

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._refresh_root_local_pos(env_ids)
        if log_episode:
            self._log_reset_episode(env_ids)

    def _refresh_root_local_pos(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        terrain_origins = self.terrain_loader.get_terrain_origin(self.env_terrain_map[env_ids])
        self.root_local_pos[env_ids] = self.root_states[env_ids, :3] - terrain_origins

    def _log_reset_episode(self, env_ids: torch.Tensor) -> None:
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # if self.cfg.terrain.curriculum:
        #     self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            # self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["command_area"] = np.sum(self.curriculum.weights) / self.curriculum.weights.shape[0]
            # self.extras["episode"]["max_vel_x"] = torch.max(self.commands[:, 0]).item()
            # self.extras["episode"]["max_vel_y"] = torch.max(self.commands[:, 1]).item()
            # self.extras["episode"]["max_vel_yaw"] = torch.max(self.commands[:, 2]).item()
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _make_reset_result(
        self,
        env_ids: torch.Tensor,
        reset_type: str,
        *,
        goal_xy: Optional[torch.Tensor] = None,
        done_mask: Optional[torch.Tensor] = None,
    ) -> ResetResult:
        terrain_ids = self.env_terrain_map[env_ids].clone()
        result_goal = goal_xy
        if result_goal is None and env_ids.numel() > 0:
            valid_goal = self.task_goal_valid[env_ids]
            if torch.all(valid_goal):
                result_goal = self.task_goal_xy[env_ids].clone()
        return ResetResult(
            env_ids=env_ids.clone(),
            reset_type=reset_type,
            terrain_ids=terrain_ids,
            start_xy=self.task_start_xy[env_ids].clone(),
            goal_xy=None if result_goal is None else result_goal.clone(),
            root_local_pos=self.root_local_pos[env_ids].clone(),
            done_mask=None if done_mask is None else done_mask.clone(),
        )

    def _run_reset_callbacks(
        self,
        reset_callbacks: Optional[Union[Callable[[ResetResult], None], Sequence[Callable[[ResetResult], None]]]],
        result: ResetResult,
    ) -> None:
        if reset_callbacks is None:
            return
        if callable(reset_callbacks):
            reset_callbacks(result)
            return
        for callback in reset_callbacks:
            callback(result)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # self.reset_buf |= self.time_out_buf
        robot_height = self.root_states[:, 2]
        fall_out = torch.where(robot_height < -5.0, True, False)
        self.reset_buf |= fall_out

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.cfg.env.randomize_reset_heading:
            yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
            base_quat = self.base_init_state[3:7].view(1, 4).expand(len(env_ids), -1)
            roll, pitch, _ = euler_from_quaternion(base_quat)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        self.render()
        for _ in range(self.cfg.control.decimation): # 4
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # terrain_info = self._get_terrain_block_info()
        # self.extras["terrain_info_x_idx"] = terrain_info["block_idx_x"]
        # self.extras["terrain_info_y_idx"] = terrain_info["block_idx_y"]
        # self.extras["base_lin_velocity"] = self.base_lin_vel
        # self.extras["base_ang_velocity"] = self.base_ang_vel
        # self.extras["command"] = self.commands
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        # Base quaternion (orientation): [qx, qy, qz, qw] - Represents robot's 3D orientation in world frame
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        terrain_origins = self.terrain_loader.get_terrain_origin(self.env_terrain_map)
        self.root_local_pos[:] = self.root_states[:, :3] - terrain_origins

        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # termination_privileged_obs = self.compute_termination_observations(env_ids)
        termination_privileged_obs = None
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.foot_contact_sensor_force = self.sensor_forces[:, self.feet_sensor_idx]
        self.contact_flag = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.foot_contact_sensor_flag = self.sensor_forces[:, self.feet_sensor_idx, 2] > 1.

        # self.disturbance[:, :, :] = 0.0

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        
        return env_ids, termination_privileged_obs

    def compute_observations(self):
        """ Computes observations by separating state from command for history. """

        # --- CHANGE 1: Create a "state-only" vector for history ---
        # This part no longer includes the command.
        imu_obs = torch.cat((self.projected_gravity, self.base_ang_vel * self.obs_scales.ang_vel), dim=-1)
        joint_obs = torch.cat(((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                               self.dof_vel * self.obs_scales.dof_vel), dim=-1)
        last_action_obs = self.actions

        # State observation (42 dims = 6 imu + 24 joint + 12 last_action)
        current_state_obs_42d = torch.cat((imu_obs, joint_obs, last_action_obs), dim=-1)
        
        # Get the command separately
        command_obs = self.commands[:, :3] * self.commands_scale

        # Ground truth for the privileged critic uses the clean state and the command.
        obs_ground_truth = torch.cat((command_obs, current_state_obs_42d), dim=-1)

        # Add noise to the state observation before it goes into history
        current_state_obs_42d_noised = current_state_obs_42d.clone()
        # if self.add_noise:
        #     # Assumes noise vector is designed for state obs (e.g., 42 dims)
        #     noise_for_state = (2 * torch.rand_like(current_state_obs_42d_noised) - 1) * self.noise_scale_vec[3:45]
        #     current_state_obs_42d_noised += noise_for_state

        # Get estimated velocity (no change here)
        v_est_for_actor = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)

        # --- CHANGE 2: The augmented observation for history is now state + velocity ---
        # This augmented state observation (45 dims = 42 state + 3 velocity) goes into history
        current_augmented_state_obs_45d = torch.cat([current_state_obs_42d_noised, v_est_for_actor], dim=-1)

        # terrain_info = self._get_terrain_block_info()
        # terrain_onehot = terrain_info["one_hot"]  # shape: [num_envs, 51]
        # combined_index = terrain_info["index"]  # shape: [num_envs, 1]
        combined_index = torch.zeros((self.num_envs), dtype=torch.int64, device=self.device)

        current_augmented_state_obs_46d = torch.cat([current_augmented_state_obs_45d, combined_index.unsqueeze(1)], dim=-1)

        # Update main history buffers with the STATE-ONLY augmented observation
        self.short_obs_history_buf = torch.cat([
            self.short_obs_history_buf[:, 1:],
            current_state_obs_42d_noised.unsqueeze(1)
        ], dim=1)

        stride = self.cfg.env.obs_history_stride
        if self.common_step_counter % stride == 0:  # or self.common_step_counter < stride:
            self.obs_history_buf = torch.cat([
                self.obs_history_buf[:, 1:],
                current_state_obs_42d_noised.unsqueeze(1)
            ], dim=1)

        # --- CHANGE 3: Reconstruct the final policy observation ---
        # The final observation for the actor now explicitly combines the current command with the current state and state histories.
        policy_obs_components = [
            command_obs,                      # Current command (3 dims)
            current_augmented_state_obs_46d,  # Current state (45 dims)
            self.short_obs_history_buf.reshape(self.num_envs, -1), # Short state history
            self.obs_history_buf.reshape(self.num_envs, -1)       # Long state history
        ]
        
        self.obs_buf = torch.cat(policy_obs_components, dim=-1)

        # Privileged observations follow the reference layout:
        # command + clean proprio + heights + privileged extras.
        if self.privileged_obs_buf is not None:
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.,
                ) * self.obs_scales.height_measurements
            else:
                num_height_points = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
                heights = torch.zeros(
                    (self.num_envs, num_height_points),
                    device=self.device,
                    dtype=torch.float,
                )

            # This Go2 setup does not use domain randomization, so keep the
            # privileged extra block at zero while preserving the expected shape.
            privileged_extras = torch.zeros(
                (self.num_envs, 42),
                device=self.device,
                dtype=torch.float,
            )

            base_lin_vel = self.base_lin_vel

            self.privileged_obs_buf = torch.cat(
                (obs_ground_truth,
                 base_lin_vel * self.obs_scales.lin_vel,
                 heights,
                 privileged_extras),
                dim=-1,
            )


import pytest


class TestEnv:

    @pytest.fixture(scope="class")
    def env(self):
        cfg = GO2RoughCfg()
        cfg.env.num_envs = 1

        from legged_gym.utils.helpers import get_args, class_to_dict, parse_sim_params
        args = get_args()

        args.sim_device = "cuda:0"
        args.headless = True

        sim_params = {"sim": class_to_dict(cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        env = GO2Rough(cfg, sim_params=sim_params, physics_engine=args.physics_engine, sim_device=args.sim_device, headless=args.headless)
        return env

    def test_init(self, env: GO2Rough):
        pass

    def test_step(self, env: GO2Rough):
        fake_action = torch.rand((env.num_envs, 12), device=env.device)
        env.step(actions=fake_action)

    def test_force_sensor(self, env: GO2Rough):
        print(f"sensor map: {env.sensor_map}")
