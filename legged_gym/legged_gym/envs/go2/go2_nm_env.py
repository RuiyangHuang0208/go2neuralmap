from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np

from legged_gym.envs.go2.go2_rough_env import GO2Rough
from legged_gym.utils.helpers import class_to_dict

try:
    from .go2_nm_config import GO2NMCfg
except ImportError:
    from legged_gym.envs.go2.go2_nm_config import GO2NMCfg
    print("Import module from absolute path for debugging")


class GO2NM(GO2Rough):
    DIRECTION_NAMES = ("front", "left", "right", "rear")

    def __init__(self, cfg: GO2NMCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.cost_scales = class_to_dict(self.cfg.cost.scales)

        self._prepare_cost_function()

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()

        self.key_body_indices = torch.zeros(len(self.key_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.key_body_names)):
            self.key_body_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.key_body_names[i])
        self.direction_body_indices = {}
        for direction_name in self.DIRECTION_NAMES:
            direction_body_names = self.direction_body_names[direction_name]
            self.direction_body_indices[direction_name] = torch.tensor(
                [
                    self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], body_name)
                    for body_name in direction_body_names
                ],
                dtype=torch.long,
                device=self.device,
            )

        # if self.gym.enable_actor_dof_force_sensors(self.envs, 0):
        #     print("Successfully enabled dof force sensors")
        # else:
        #     print("Unable to enabled dof force sensors, use pid calculating torque")

        self.step_cost = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.dof_force = gymtorch.wrap_tensor(dof_force_tensor)

    def _add_force_sensors(self, robot_asset):
        super()._add_force_sensors(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        key_body_names = self._expand_body_name_patterns(self.cfg.cost.key_bodies, body_names, "cost.key_bodies")
        sensor_pose = gymapi.Transform()
        key_body_sensor_idx = []
        self.key_body_names = key_body_names
        self.direction_body_names = {
            direction_name: self._expand_body_name_patterns(
                self.cfg.cost.body_direction_group.get(direction_name, []),
                body_names,
                f"cost.body_direction_group[{direction_name}]",
            )
            for direction_name in self.DIRECTION_NAMES
        }

        for name in key_body_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = True  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            sensor_options.use_world_frame = True  # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            sensor_idx = self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
            self.sensor_map[name] = sensor_idx
            key_body_sensor_idx.append(sensor_idx)
        self.key_body_sensor_idx = torch.tensor(key_body_sensor_idx, dtype=torch.long, device=self.device)

    @staticmethod
    def _expand_body_name_patterns(patterns, body_names, cfg_name):
        expanded_body_names = []
        for pattern in patterns:
            matched_body_names = [body_name for body_name in body_names if pattern in body_name]
            if len(matched_body_names) == 0:
                raise ValueError(f"No rigid body matched pattern '{pattern}' in {cfg_name}.")
            for body_name in matched_body_names:
                if body_name not in expanded_body_names:
                    expanded_body_names.append(body_name)
        return expanded_body_names

    def _prepare_cost_function(self):

        for key in list(self.cost_scales.keys()):
            scale = self.cost_scales[key]
            if scale == 0:
                self.cost_scales.pop(key)
            else:
                # self.cost_scales[key] *= self.dt
                pass

        self.cost_functions = []
        self.cost_names = []
        for name, scale in self.cost_scales.items():
            if name == "termination":
                continue
            self.cost_names.append(name)
            name = f"_cost_{name}"
            self.cost_functions.append(getattr(self, name))

    def compute_cost(self):
        self.step_cost[:] = 0.0
        for i in range(len(self.cost_functions)):
            name = self.cost_names[i]
            cost = self.cost_functions[i]() * self.cost_scales[name]
            self.step_cost += cost
        if "termination" in self.cost_scales:
            cost = self._cost_termination() * self.cost_scales["termination"]
            self.step_cost += cost

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()


        self.gym.refresh_dof_force_tensor(self.sim)
        self.compute_cost()


    def _cost_power(self):
        dof_vel = self.dof_vel
        # torque = self.dof_force
        torque = self.torques
        power: torch.Tensor = dof_vel * torque

        # root_vel: torch.Tensor = self.root_states[:, 7:10]
        # mass = self.cfg.asset.robot_mass
        # gravity_acc = 9.81 # [m/s^2]
        # cost = power.abs().sum(dim=-1) / (mass * gravity_acc * root_vel.norm(dim=-1))
        cost = power.abs().sum(dim=-1)
        # print(f"dof vel: {dof_vel}")
        # print(f"dof force: {torque}")
        # print(f"pow cost: {cost}")
        return cost

    def _cost_collision(self):
        # key_body_contact_sensor_force = self.sensor_forces[:, self.key_body_sensor_idx]
        key_body_contact_sensor_force = self.contact_forces[:, self.key_body_indices]
        # key_body_contact_flag = self.contact_forces[:, self.key_body_indices].norm(dim=-1).max(dim=-1) > 5.
        contact_force = key_body_contact_sensor_force.norm(dim=-1).max(dim=-1).values
        # contact_force = key_body_contact_sensor_force.norm(dim=-1).sum(dim=-1)
        # print(f"contact force {contact_force}")
        key_body_contact_sensor_flag = contact_force > self.cfg.cost.collision_contact_threshold

        # return key_body_contact_sensor_flag.to(torch.float)
        return contact_force
        # return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def get_directional_collision_flags(self):
        directional_collision_flags = []
        threshold = self.cfg.cost.collision_contact_threshold
        for direction_name in self.DIRECTION_NAMES:
            direction_indices = self.direction_body_indices[direction_name]
            if len(direction_indices) == 0:
                direction_collision_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            else:
                direction_contact_force = self.contact_forces[:, direction_indices].norm(dim=-1).max(dim=-1).values
                direction_collision_flag = direction_contact_force > threshold
            directional_collision_flags.append(direction_collision_flag)
        return torch.stack(directional_collision_flags, dim=-1)

    def _draw_debug_vis(self):
        super()._draw_debug_vis()

        self.draw_collision_contact_markers(clear_lines=not self.terrain.cfg.measure_heights)

    def draw_collision_contact_markers(self, clear_lines=False):
        if not self.cfg.cost.visualize_collision_contacts:
            return

        if clear_lines:
            self.gym.clear_lines(self.viewer)

        threshold = self.cfg.cost.collision_contact_threshold
        key_body_force_norm = self.contact_forces[0, self.key_body_indices].norm(dim=-1)
        contact_body_ids = (key_body_force_norm > threshold).nonzero(as_tuple=False).flatten()
        if len(contact_body_ids) == 0:
            return

        body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        key_body_pos = body_states[0, self.key_body_indices[contact_body_ids], 0:3].cpu().numpy()
        marker_geom = gymutil.WireframeSphereGeometry(
            self.cfg.cost.collision_contact_viz_marker_radius,
            8,
            8,
            None,
            color=(1, 0, 0),
        )

        for pos in key_body_pos:
            marker_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
            gymutil.draw_lines(marker_geom, self.gym, self.viewer, self.envs[0], marker_pose)

    def _cost_termination(self):
        # return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # NOTE: Currently there's no time termination. If there is, here should be changed
        return self.reset_buf.clone() * 100


import pytest


class TestEnv:

    @pytest.fixture(scope="class")
    def env(self):
        cfg = GO2NMCfg()
        cfg.env.num_envs = 1
        cfg.cost.scales.power = 0.0

        from legged_gym.utils.helpers import get_args, class_to_dict, parse_sim_params
        args = get_args()

        args.sim_device = "cuda:0"
        args.headless = True

        sim_params = {"sim": class_to_dict(cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        env = GO2NM(cfg, sim_params=sim_params, physics_engine=args.physics_engine, sim_device=args.sim_device, headless=args.headless)
        return env

    def test_init(self, env: GO2NM):
        pass

    def test_step(self, env: GO2NM):
        fake_action = torch.rand((env.num_envs, 12), device=env.device)
        env.step(actions=fake_action)
        directional_collision_flags = env.get_directional_collision_flags()
        assert directional_collision_flags.shape == (env.num_envs, len(env.DIRECTION_NAMES))
        assert directional_collision_flags.dtype == torch.bool

    def test_cost_calc(self, env: GO2NM):

        step_cost = env.step_cost
        print(f"[DEBUG][TIMESTEP=0]:{step_cost}")

        fake_action = torch.rand((env.num_envs, 12), device=env.device)
        env.step(actions=fake_action)

        step_cost = env.step_cost
        print(f"[DEBUG][TIMESTEP=1]:{step_cost}")
