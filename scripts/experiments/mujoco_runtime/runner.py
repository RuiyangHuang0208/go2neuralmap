from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

from builder import (
    build_cost_estimator,
    build_cost_map,
    build_locomotion_model,
    build_neural_map,
    build_path_planner,
    build_position_controller,
    build_state_recorder,
    build_vel_cmd_generator,
)
from modules.vel_command_generator.vel_command_generator import normalize_vel_cmd_shape

from .env_adapter import MuJoCoGo2EnvAdapter, MuJoCoGo2EnvConfig
from .observation import Go2LocomotionObservationBuilder


RUNNER_LOGGER = logger.bind(module_name="MuJoCoEvaluation")
HEADING_TARGET_COMMAND_GENERATORS = {
    "keyboard_input",
    "point_to_point_navigation",
    "position_navigation",
}


@dataclass
class MuJoCoEvaluationConfig:
    episodes: int = 1
    max_steps: int = 2500
    command_mode: str = "fixed"
    fixed_command: tuple[float, float, float] = (0.4, 0.0, 0.0)
    update_neural_map: bool = True
    output: Optional[str] = None


class FixedCommandGenerator:
    def __init__(self, env, command: tuple[float, float, float]):
        self.env = env
        self.command = torch.tensor(command, dtype=torch.float32, device=env.device).view(1, 3)
        self.info: Dict[str, Any] = {}

    def init_state(self, **kwargs) -> None:
        self.info = {}

    def compute_vel_cmd(self, **kwargs) -> torch.Tensor:
        command = torch.zeros_like(self.env.commands)
        command[:, :3] = self.command
        self.info = {"command": self.command[0].detach().cpu()}
        return command

    def retrieve_markers(self):
        return []

    def get_info(self):
        return self.info


class MuJoCoEvaluationRunner:
    def __init__(self, cfg_omega, mujoco_cfg: MuJoCoGo2EnvConfig, eval_cfg: MuJoCoEvaluationConfig, terrain_loader=None):
        self.cfg_omega = cfg_omega
        self.mujoco_cfg = mujoco_cfg
        self.eval_cfg = eval_cfg
        self.env = MuJoCoGo2EnvAdapter(mujoco_cfg, terrain_loader=terrain_loader)
        self.env.set_fixed_goal()
        self.obs_builder = Go2LocomotionObservationBuilder(self.env.go2_cfg, self.env.device)

        self.cfg_omega.modules.neural_map.num_envs = self.env.num_envs
        if OmegaConf.select(self.cfg_omega, "modules.neural_map.config.device", default=None) is not None:
            self.cfg_omega.modules.neural_map.config.device = str(self.env.device)
        self.cfg_omega.modules.cost_map.num_envs = self.env.num_envs
        if OmegaConf.select(self.cfg_omega, "modules.cost_map.config.device", default=None) is not None:
            self.cfg_omega.modules.cost_map.config.device = str(self.env.device)
        self.neural_map = build_neural_map(self.cfg_omega.modules.neural_map)
        self.cost_map = build_cost_map(self.cfg_omega.modules.cost_map)
        self.locomotion_model = build_locomotion_model(self.cfg_omega.modules.locomotion_model).to(self.env.device)
        self.cost_estimator = build_cost_estimator(self.cfg_omega.modules.cost_estimator, self.env)
        self.position_controller = build_position_controller(self.cfg_omega.modules.position_controller)
        self.planner = build_path_planner(self.cfg_omega.modules.planner)
        self.vel_cmd_generator = self._build_vel_cmd_generator()
        self.state_recorder = build_state_recorder(self.cfg_omega.helpers.state_recorder)
        self.viewer = None

    def _build_vel_cmd_generator(self):
        if self.eval_cfg.command_mode == "fixed" or str(self.cfg_omega.modules.vel_command_generator.name) == "null_impl":
            return FixedCommandGenerator(self.env, tuple(float(v) for v in self.eval_cfg.fixed_command))
        return build_vel_cmd_generator(self.cfg_omega.modules.vel_command_generator, self)

    def _convert_heading_target_command(self, vel_cmd: torch.Tensor) -> torch.Tensor:
        generator_name = str(self.cfg_omega.modules.vel_command_generator.name)
        if generator_name not in HEADING_TARGET_COMMAND_GENERATORS:
            return vel_cmd
        if vel_cmd.shape[-1] <= 3:
            return vel_cmd

        converted = vel_cmd.clone()
        heading_error = self._wrap_to_pi(converted[:, 3] - self.env.yaw)
        converted[:, 2] = torch.clamp(0.5 * heading_error, min=-1.0, max=1.0).to(dtype=converted.dtype)
        return converted

    def run(self) -> Dict[str, Any]:
        self.vel_cmd_generator.init_state()
        obs = self.obs_builder.build(self.env)
        completed_episodes = 0
        success_count = 0
        timeout_count = 0
        min_distance = float("inf")
        last_render_sync_time = time.time()

        viewer_ctx = (
            mujoco.viewer.launch_passive(self.env.model, self.env.data, show_left_ui=False, show_right_ui=False)
            if self.mujoco_cfg.render
            else None
        )
        self.viewer = viewer_ctx
        if self.viewer is not None:
            self.viewer.cam.lookat[:] = [1.0, 0.0, 0.2]
            self.viewer.cam.distance = 4.0
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20

        try:
            with torch.no_grad():
                for step in range(int(self.eval_cfg.max_steps)):
                    step_start_time = time.time()
                    vel_cmd = normalize_vel_cmd_shape(
                        self.vel_cmd_generator.compute_vel_cmd(),
                        self.env.commands.shape,
                    )
                    vel_cmd = self._convert_heading_target_command(vel_cmd)
                    self.env.commands[:, :] = vel_cmd
                    self.locomotion_model.compute_latents(obs.detach())
                    actions = self.locomotion_model.act()
                    self.env.step(actions.detach())
                    obs = self.obs_builder.build(self.env)
                    privileged_obs = self.obs_builder.build_privileged(self.env)

                    if bool(self.eval_cfg.update_neural_map):
                        self.neural_map.update(
                            long_latents=self.locomotion_model.out["latent_long"].detach().clone(),
                            short_latents=self.locomotion_model.out["short_proj"].detach().clone(),
                            xy_coord=self.env.root_local_pos[:, :2],
                            root_velocity=self.env.root_states[:, 7:9],
                        )
                    long_latent = self.locomotion_model.out["latent_long"].detach()
                    short_latent = self.locomotion_model.out["short_proj"].detach()
                    latent = torch.cat([long_latent, short_latent], dim=-1)
                    cost = self.cost_estimator.compute_cost(
                        latent=latent,
                        privileged_obs=privileged_obs,
                        neural_map=self.neural_map,
                    )
                    self.cost_map.update(
                        current_pos=self.env.root_local_pos[:, :2],
                        new_values=cost.detach().clone(),
                        root_velocity=self.env.root_states[:, 7:9],
                        command=self.env.commands[:, :2],
                        directional_collision=self.env.get_directional_collision_flags(),
                        base_quat=self.env.base_quat,
                        latent=latent,
                    )

                    info = self._augment_navigation_info(self.vel_cmd_generator.get_info())
                    distance = self._extract_distance(info)
                    if distance is not None:
                        min_distance = min(min_distance, float(distance))
                    target_episodes = int(self.eval_cfg.episodes)
                    if bool(info.get("task_terminal_flag", False)):
                        completed_episodes += 1
                        if bool(info.get("task_success_flag", False)):
                            success_count += 1
                        if bool(info.get("task_timeout_flag", False)):
                            timeout_count += 1
                        if target_episodes > 0 and completed_episodes >= target_episodes:
                            break

                    self._record_state(step, info, cost=cost)
                    if self.viewer is not None:
                        if not self.viewer.is_running():
                            break
                        render_dt = 1.0 / max(float(self.mujoco_cfg.render_fps), 1.0)
                        if time.time() - last_render_sync_time >= render_dt:
                            self._update_task_marker_geoms(info)
                            self.viewer.sync()
                            last_render_sync_time = time.time()
                    if self.mujoco_cfg.realtime:
                        time.sleep(max(0.0, self.env.dt - (time.time() - step_start_time)))
        finally:
            if self.viewer is not None:
                self.viewer.close()
            self.state_recorder.save()

        summary = {
            "completed_episodes": completed_episodes,
            "success_count": success_count,
            "timeout_count": timeout_count,
            "success_rate": success_count / completed_episodes if completed_episodes > 0 else 0.0,
            "min_distance_to_goal": None if min_distance == float("inf") else min_distance,
            "steps": self.env.common_step_counter,
            "control_dt": self.env.dt,
            "xml_path": self.mujoco_cfg.xml_path,
        }
        if self.eval_cfg.output:
            output_path = Path(self.eval_cfg.output).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            RUNNER_LOGGER.info("Saved MuJoCo evaluation summary to {}", output_path)
        return summary

    def _record_state(self, step: int, info: Dict[str, Any], *, cost: torch.Tensor) -> None:
        uncertainty_map_vals = None
        if hasattr(self.neural_map, "uncertainties"):
            uncertainty_source = (
                self.neural_map.get_aged_uncertainties()
                if hasattr(self.neural_map, "get_aged_uncertainties")
                else self.neural_map.uncertainties
            )
            if uncertainty_source.dim() == 4 and uncertainty_source.shape[-1] == 4:
                uncertainty_map_vals = uncertainty_source[0].permute(2, 0, 1).detach().cpu()

        self.state_recorder.collect_vars(
            {
                "step": step,
                "current_xy": self.env.root_local_pos[0, :2].detach().cpu(),
                "root_states": self.env.root_states[0].detach().cpu(),
                "commands": self.env.commands[0].detach().cpu(),
                "cost_map_vals": self.cost_map.maps[0].detach().cpu(),
                "vel_cmd_generator_info": info,
                "reset_buf": self.env.reset_buf.detach().cpu(),
                "collision_flags": self.env.get_directional_collision_flags().detach().cpu(),
                "dt": float(self.env.dt),
                "current_cost": cost.detach().cpu(),
                "uncertainty_map_vals": uncertainty_map_vals,
                "uncertainty_map_metadata": {
                    "x_range": tuple(float(value) for value in self.neural_map.cfg.x_range),
                    "y_range": tuple(float(value) for value in self.neural_map.cfg.y_range),
                    "least_count": float(self.neural_map.cfg.least_count),
                },
            }
        )
        self.state_recorder.handle()

    def _augment_navigation_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info or {})
        info.setdefault("current_xy", self.env.root_local_pos[0, :2].detach().cpu())
        if self.env.task_goal_valid[0]:
            info.setdefault("goal_xy", self.env.task_goal_xy[0].detach().cpu())
            distance = torch.norm(self.env.root_local_pos[0, :2] - self.env.task_goal_xy[0])
            info.setdefault("distance_to_goal", distance.detach().cpu())
        return info

    @staticmethod
    def _extract_distance(info: Dict[str, Any]) -> Optional[float]:
        value = info.get("distance_to_goal")
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.flatten()[0].item())
        return float(value)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi

    def _update_task_marker_geoms(self, info: Dict[str, Any]) -> None:
        if self.viewer is None:
            return
        start_xy = self._marker_xy_from_info(info.get("task_start_xy"))
        if start_xy is None and hasattr(self.env, "last_reset_local_pos"):
            start_xy = self.env.last_reset_local_pos[0, :2].detach()
        goal_xy = self._marker_xy_from_info(info.get("goal_xy"))
        if goal_xy is None and bool(self.env.task_goal_valid[0]):
            goal_xy = self.env.task_goal_xy[0].detach()

        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            if start_xy is not None:
                self._add_marker_geom(
                    label="start",
                    xy=start_xy,
                    radius=0.08,
                    rgba=(0.1, 0.9, 0.2, 0.85),
                )
            if goal_xy is not None:
                self._add_marker_geom(
                    label="goal",
                    xy=goal_xy,
                    radius=0.1,
                    rgba=(0.0, 0.85, 1.0, 0.85),
                )

    def _add_marker_geom(
        self,
        *,
        label: str,
        xy: torch.Tensor,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        scene = self.viewer.user_scn
        if scene.ngeom >= len(scene.geoms):
            RUNNER_LOGGER.warning("MuJoCo user scene geom capacity reached; skipping {} marker.", label)
            return
        pos = self._local_xy_to_marker_world_pos(xy, z_offset=float(radius) + 0.03)
        size = np.array([float(radius), 0.0, 0.0], dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(-1)
        rgba_array = np.asarray(rgba, dtype=np.float32)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            pos,
            mat,
            rgba_array,
        )
        scene.ngeom += 1

    def _local_xy_to_marker_world_pos(self, xy: torch.Tensor, *, z_offset: float) -> np.ndarray:
        xy = torch.as_tensor(xy, dtype=torch.float32, device=self.env.device).flatten()[:2]
        if self.env.terrain_loader is None:
            return np.asarray([float(xy[0].item()), float(xy[1].item()), float(z_offset)], dtype=np.float64)

        terrain_ids = self.env.env_terrain_map[:1].to(device=self.env.device, dtype=torch.long)
        ground_z = self.env.terrain_loader.get_height(terrain_ids[0], xy).to(
            device=self.env.device,
            dtype=torch.float32,
        )
        local_pos = torch.stack((xy[0], xy[1], ground_z.flatten()[0] + float(z_offset))).view(1, 3)
        world_pos = self.env.terrain_loader.local_to_world(terrain_ids, local_pos)[0]
        return world_pos.detach().cpu().numpy().astype(np.float64)

    @staticmethod
    def _marker_xy_from_info(value: Any) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
        if tensor.numel() < 2:
            return None
        return tensor[:2]
