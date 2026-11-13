from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_apply
from loguru import logger
from omegaconf import OmegaConf

from utils.diffusion_policy.dataset import (
    crop_directional_map,
    crop_uncertainty_map,
    gather_directional_latent,
    training_target_to_commands,
)
from utils.diffusion_policy.registry import build_algorithm
from utils.position_controller import grid_to_xy

from .vel_command_generator import VelCommandGenerator, VelCommandGeneratorConfig


@dataclass
class DiffusionPolicyVelCommandGeneratorConfig(VelCommandGeneratorConfig):
    run_dir: str = "logs/diffusion_policy/ddpm_2026-05-01_09-27-28"
    checkpoint: Optional[str] = None
    config: Optional[str] = None
    device: str = "cuda"
    execution_horizon: int = 2
    command_dt: float = 0.5
    main_loop_hz: float = 50.0
    goal_tolerance: float = 0.35
    resample_goal_flag: bool = True
    reset_env_flag: bool = False
    goal_source: str = "sampled"
    visualize_prediction_path: bool = False
    prediction_marker_radius: float = 0.06
    prediction_marker_height_offset: float = 0.08


@dataclass
class DiffusionPolicyRunningInfo:
    current_xy: Optional[torch.Tensor] = None
    current_grid_xy: Optional[Tuple[int, int]] = None
    goal_xy: Optional[torch.Tensor] = None
    sequence_index: int = 0
    command_step_index: int = 0
    sampled_sequence: bool = False
    reached_goal_flag: bool = False


class DiffusionPolicyVelCommandGenerator(VelCommandGenerator):
    def __init__(self, cfg: DiffusionPolicyVelCommandGeneratorConfig, context=None, **kwargs):
        self.logger = logger.bind(module_name="VelCmdGen.DiffusionPolicy")
        if context is None:
            raise ValueError("No context is passed")

        self.cfg = cfg
        self.context = context
        self.env = context.env
        self.markers = []
        self.info: Dict = {}
        if cfg.goal_source not in ("sampled", "user"):
            raise ValueError(f"Unknown goal_source: {cfg.goal_source}")

        self.run_dir = Path(cfg.run_dir).expanduser()
        self.config_path = Path(cfg.config).expanduser() if cfg.config is not None else self.run_dir / "config.yaml"
        self.checkpoint_path = (
            Path(cfg.checkpoint).expanduser() if cfg.checkpoint is not None else self.run_dir / "latest.pt"
        )
        self.device = self._resolve_device(cfg.device)
        self.cfg_omega = self._load_config()
        self.algorithm = self._load_algorithm()
        self.algorithm.model.eval()

        dataset_cfg = self.cfg_omega.training.dataset
        self.crop_size = int(dataset_cfg.crop_size)
        self.goal_scale = float(dataset_cfg.goal_scale)
        self.velocity_scale = float(dataset_cfg.velocity_scale)
        self.uncertainty_log_scale = float(dataset_cfg.uncertainty_log_scale)
        self.uncertainty_max = float(dataset_cfg.uncertainty_max)
        self.model_horizon = int(self.cfg_omega.algorithm.config.model.horizon)
        self.command_dim = int(self.cfg_omega.algorithm.config.model.command_dim)
        self.execution_horizon = min(int(cfg.execution_horizon), self.model_horizon)
        if self.execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        self.steps_per_command = max(1, int(round(float(cfg.command_dt) * float(cfg.main_loop_hz))))

        self.goal_xy: Optional[torch.Tensor] = None
        self.user_goal_xy: Optional[torch.Tensor] = None
        self.command_sequence = torch.zeros((0, 3), dtype=torch.float32, device=self.env.device)
        self.prediction_sequence = torch.zeros((0, 3), dtype=torch.float32, device=self.env.device)
        self.sequence_index = 0
        self.command_step_index = 0

        self.logger.info(
            "Loaded diffusion policy checkpoint={} model_horizon={} execution_horizon={} steps_per_command={}",
            self.checkpoint_path,
            self.model_horizon,
            self.execution_horizon,
            self.steps_per_command,
        )

    def set_goal_xy(self, goal_xy: torch.Tensor) -> None:
        self.user_goal_xy = torch.as_tensor(goal_xy, dtype=torch.float32, device=self.env.device).view(2)
        self.goal_xy = self.user_goal_xy.clone()
        self._invalidate_sequence()

    def clear_user_goal(self) -> None:
        self.user_goal_xy = None
        self.goal_xy = None
        self._invalidate_sequence()

    def init_state(self, **kwargs):
        self._refresh_goal(resample=True)
        self._invalidate_sequence()

    @torch.no_grad()
    def compute_vel_cmd(self, **kwargs):
        info_struct = DiffusionPolicyRunningInfo()
        current_xy = self.env.root_local_pos[0, :2]
        current_grid_xy = self._xy_to_grid(current_xy)
        info_struct.current_xy = current_xy.detach().cpu()
        info_struct.current_grid_xy = current_grid_xy

        reset_buf = getattr(self.env, "reset_buf", None)
        if reset_buf is not None and bool(reset_buf[0].item()):
            self.logger.info("[Reset] Environment reset detected. Reinitializing diffusion navigation state.")
            self._refresh_goal(resample=self.cfg.resample_goal_flag)
            self._invalidate_sequence()

        if self.goal_xy is None:
            self._refresh_goal(resample=True)

        assert self.goal_xy is not None
        distance_to_goal = torch.norm(current_xy - self.goal_xy)
        if distance_to_goal <= float(self.cfg.goal_tolerance):
            info_struct.reached_goal_flag = True
            if self.cfg.reset_env_flag:
                if hasattr(self.env, "task_reset"):
                    result = self.env.task_reset(
                        torch.tensor([0], dtype=torch.long, device=self.env.device),
                        resample_goal=self.cfg.resample_goal_flag,
                    )
                    if result.goal_xy is not None:
                        self.goal_xy = result.goal_xy[0].to(device=self.env.device, dtype=torch.float32)
                else:
                    self.env.reset()
                    self._refresh_goal(resample=self.cfg.resample_goal_flag)
                current_xy = self.env.root_local_pos[0, :2]
            if not self.cfg.reset_env_flag:
                self._refresh_goal(resample=self.cfg.resample_goal_flag)
            self._invalidate_sequence()

        if self._needs_new_sequence():
            self.command_sequence = self._sample_command_sequence()
            self.sequence_index = 0
            self.command_step_index = 0
            info_struct.sampled_sequence = True

        env_cmd = self._current_env_command()
        self._advance_sequence_cursor()
        self._update_markers(current_xy)

        info_struct.goal_xy = self.goal_xy.detach().cpu()
        info_struct.sequence_index = self.sequence_index
        info_struct.command_step_index = self.command_step_index
        self.info = asdict(info_struct)
        self.info["model_horizon"] = self.model_horizon
        self.info["execution_horizon"] = self.execution_horizon
        self.info["steps_per_command"] = self.steps_per_command
        self.info["goal_source"] = self.cfg.goal_source
        return env_cmd

    def retrieve_markers(self) -> List:
        return self.markers

    def get_info(self) -> Dict:
        return self.info

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Diffusion policy config not found: {self.config_path}")
        return OmegaConf.load(self.config_path)

    def _load_algorithm(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Diffusion policy checkpoint not found: {self.checkpoint_path}")
        algorithm = build_algorithm(self.cfg_omega.algorithm).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        algorithm_state = checkpoint.get("algorithm", checkpoint)
        if "model" not in algorithm_state:
            raise KeyError(f"Checkpoint {self.checkpoint_path} does not contain algorithm.model state")
        algorithm.model.load_state_dict(algorithm_state["model"])
        return algorithm

    def _resolve_device(self, requested: str) -> torch.device:
        if requested in ("gpu", "cuda"):
            requested = "cuda"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.logger.warning("Requested {} but CUDA is unavailable; using CPU", requested)
            return torch.device("cpu")
        return torch.device(requested)

    def _refresh_goal(self, *, resample: bool) -> None:
        if self.user_goal_xy is not None:
            self.goal_xy = self.user_goal_xy.clone()
            return
        if self.goal_xy is None or resample:
            if hasattr(self.env, "resample_goals"):
                result = self.env.resample_goals(torch.tensor([0], dtype=torch.long, device=self.env.device))
                assert result.goal_xy is not None
                self.goal_xy = result.goal_xy[0].detach().clone()
            else:
                self.goal_xy = self.env.sample_goals(0)[0, :2].detach().clone()
            self.logger.info("Sampled diffusion navigation goal at: {}", self.goal_xy)

    def _invalidate_sequence(self) -> None:
        self.command_sequence = torch.zeros((0, 3), dtype=torch.float32, device=self.env.device)
        self.prediction_sequence = torch.zeros((0, 3), dtype=torch.float32, device=self.env.device)
        self.sequence_index = 0
        self.command_step_index = 0

    def _needs_new_sequence(self) -> bool:
        return (
            self.command_sequence.numel() == 0
            or self.sequence_index >= min(self.execution_horizon, self.command_sequence.shape[0])
        )

    def _current_env_command(self) -> torch.Tensor:
        env_cmd = torch.zeros(4, dtype=torch.float32, device=self.env.device)
        if self.command_sequence.numel() == 0:
            env_cmd[3] = self._compute_current_heading()
            return env_cmd
        cmd = self.command_sequence[self.sequence_index]
        env_cmd[:2] = cmd[:2]
        env_cmd[3] = cmd[2]
        return env_cmd

    def _advance_sequence_cursor(self) -> None:
        self.command_step_index += 1
        if self.command_step_index >= self.steps_per_command:
            self.command_step_index = 0
            self.sequence_index += 1

    def _sample_command_sequence(self) -> torch.Tensor:
        batch = self._build_inference_batch()
        if hasattr(self.algorithm, "infer"):
            target = self.algorithm.infer(batch, self.model_horizon)
        elif hasattr(self.algorithm, "sample"):
            target = self.algorithm.sample(batch, self.model_horizon)
        else:
            target = self.algorithm.model(batch)
        commands = training_target_to_commands(target, velocity_scale=self.velocity_scale)[0]
        commands[:, 2] = torch.remainder(commands[:, 2] + math.pi, 2.0 * math.pi) - math.pi
        self.prediction_sequence = commands.detach().to(self.env.device)
        return commands[: self.execution_horizon].detach().to(self.env.device)

    def _build_inference_batch(self) -> Dict[str, torch.Tensor]:
        assert self.goal_xy is not None
        neural_map = self.context.neural_map
        current_xy = self.env.root_local_pos[0, :2]
        start_grid = torch.tensor(self._xy_to_grid(current_xy), dtype=torch.long, device=self.env.device)
        start_xy = grid_to_xy(
            (int(start_grid[0].item()), int(start_grid[1].item())),
            neural_map.cfg.x_range,
            neural_map.cfg.y_range,
            neural_map.cfg.least_count,
            self.env.device,
        )

        map_tensor = neural_map.maps[0].detach().to(device=self.device, dtype=torch.float32)
        if hasattr(neural_map, "get_aged_uncertainties"):
            uncertainty_source = neural_map.get_aged_uncertainties()
        else:
            uncertainty_source = neural_map.uncertainties
        uncertainty_tensor = uncertainty_source[0].detach().to(device=self.device, dtype=torch.float32)
        start_grid_device = start_grid.to(self.device)

        map_crop = crop_directional_map(
            map_tensor,
            start_grid_device,
            crop_size=self.crop_size,
            pad_value=0.0,
        ).unsqueeze(0)
        uncertainty_crop = crop_uncertainty_map(
            uncertainty_tensor,
            start_grid_device,
            crop_size=self.crop_size,
            pad_value=self.uncertainty_max,
        )
        uncertainty_crop = torch.log1p(torch.clamp(uncertainty_crop, min=0.0))
        uncertainty_crop = torch.clamp(
            uncertainty_crop / self.uncertainty_log_scale,
            min=0.0,
            max=1.0,
        ).unsqueeze(0)
        current_latent = gather_directional_latent(map_tensor, start_grid_device).unsqueeze(0)
        goal_delta = ((self.goal_xy.to(self.device) - start_xy.to(self.device)) / self.goal_scale).view(1, 2)
        current_heading = self._compute_current_heading().to(self.device)
        # current_heading_features = torch.stack([torch.sin(current_heading), torch.cos(current_heading)]).view(1, 2)
        current_heading_features = torch.stack([torch.sin(current_heading), torch.cos(current_heading)]).view(1, 2)
        dummy_target = torch.zeros((1, self.model_horizon, self.command_dim), dtype=torch.float32, device=self.device)
        dummy_mask = torch.ones((1, self.model_horizon), dtype=torch.bool, device=self.device)

        return {
            "map_crop": map_crop,
            "uncertainty_crop": uncertainty_crop,
            "goal_delta": goal_delta,
            "current_latent": current_latent,
            "current_heading": current_heading_features,
            "target": dummy_target,
            "target_mask": dummy_mask,
        }

    def _xy_to_grid(self, xy: torch.Tensor) -> Tuple[int, int]:
        cfg = self.context.neural_map.cfg
        x_idx = int(torch.floor((xy[0] - float(cfg.x_range[0])) / float(cfg.least_count)).item())
        y_idx = int(torch.floor((xy[1] - float(cfg.y_range[0])) / float(cfg.least_count)).item())
        h, w = self.context.neural_map.maps.shape[1:3]
        return (max(0, min(x_idx, h - 1)), max(0, min(y_idx, w - 1)))

    def _compute_current_heading(self) -> torch.Tensor:
        root_quat_xyzw = self.env.root_states[0, 3:7]
        forward_vec = root_quat_xyzw.new_tensor([[1.0, 0.0, 0.0]])
        forward_dir = quat_apply(root_quat_xyzw.view(1, 4), forward_vec)
        return torch.atan2(forward_dir[:, 1], forward_dir[:, 0]).squeeze(0)

    def _update_markers(self, current_xy: torch.Tensor) -> None:
        if self.goal_xy is None:
            self.markers = []
            return
        terrain_id = self.env.env_terrain_map[0]
        terrain_origin = self.env.terrain_loader._instances[terrain_id].array_offset
        robot_marker = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
        goal_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 1))
        robot_h = self.env.terrain_loader.get_height(terrain_id, current_xy)
        goal_h = self.env.terrain_loader.get_height(terrain_id, self.goal_xy)
        robot_pose = gymapi.Transform(
            gymapi.Vec3(current_xy[0].item() + terrain_origin[0], current_xy[1].item() + terrain_origin[1], robot_h),
            r=None,
        )
        goal_pose = gymapi.Transform(
            gymapi.Vec3(self.goal_xy[0].item() + terrain_origin[0], self.goal_xy[1].item() + terrain_origin[1], goal_h),
            r=None,
        )
        self.markers = [(robot_marker, robot_pose), (goal_marker, goal_pose)]
        if bool(self.cfg.visualize_prediction_path):
            self.markers.extend(
                self._build_prediction_path_markers(
                    current_xy=current_xy,
                    terrain_id=terrain_id,
                    terrain_origin=terrain_origin,
                )
            )

    def _build_prediction_path_markers(
        self,
        *,
        current_xy: torch.Tensor,
        terrain_id: int,
        terrain_origin,
    ) -> List[Tuple[gymutil.WireframeSphereGeometry, gymapi.Transform]]:
        if self.prediction_sequence.numel() == 0:
            return []

        marker = gymutil.WireframeSphereGeometry(
            float(self.cfg.prediction_marker_radius),
            4,
            4,
            None,
            color=(1.0, 0.45, 0.0),
        )
        positions = self._integrate_prediction_positions(current_xy, self.prediction_sequence)
        markers = []
        for pos_xy in positions[1:]:
            marker_h = (
                self.env.terrain_loader.get_height(terrain_id, pos_xy)
                + float(self.cfg.prediction_marker_height_offset)
            )
            marker_pose = gymapi.Transform(
                gymapi.Vec3(
                    pos_xy[0].item() + terrain_origin[0],
                    pos_xy[1].item() + terrain_origin[1],
                    marker_h,
                ),
                r=None,
            )
            markers.append((marker, marker_pose))
        return markers

    def _integrate_prediction_positions(self, current_xy: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        dt = float(self.cfg.command_dt)
        positions = [current_xy.detach().to(device=self.env.device, dtype=torch.float32).clone()]
        pos = positions[0].clone()
        for command in commands.detach().to(device=self.env.device, dtype=torch.float32):
            vx = command[0]
            vy = command[1]
            heading = command[2]
            if not bool(torch.isfinite(heading).item()):
                heading = self._compute_current_heading().to(device=self.env.device)
            cos_h = torch.cos(heading)
            sin_h = torch.sin(heading)
            delta = torch.stack([
                cos_h * vx - sin_h * vy,
                sin_h * vx + cos_h * vy,
            ]) * dt
            pos = pos + delta
            positions.append(pos.clone())
        return torch.stack(positions)
