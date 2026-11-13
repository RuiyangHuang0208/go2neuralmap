from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from loguru import logger

from utils.diffusion_policy.dataset import (
    crop_cost_map,
    crop_directional_map,
    crop_uncertainty_map,
    gather_directional_latent,
    scale_uncertainty_crop,
)
from utils.neural_map import DirectionalBayesianNeuralMap


@dataclass
class DatasetCollectorConfig:
    enabled: bool = False
    name: str = "snapshot"
    output_dir: str = ""
    sample_interval: int = 1
    shard_size: int = 256
    dtype: str = "float32"
    env_id: int = 0
    velocity_horizon: int = 10
    velocity_command_dt: float = 0.5
    main_loop_hz: float = 50.0
    drop_incomplete_horizon: bool = False
    max_path_length: int = 256
    max_waypoints: int = 64
    crop_size: int = 8
    uncertainty_log_scale: float = 12.0
    uncertainty_max: float = 100000.0
    cost_obstacle_threshold: float = 70.0
    robot_radius_cells: int = 1


class NullDatasetCollector:
    enabled = False

    def validate(self, *args, **kwargs) -> None:
        return

    def collect(self, *args, **kwargs) -> None:
        return

    def flush(self) -> None:
        return


class DatasetCollector:
    """Shard full-map snapshots collected from the running starter pipeline."""

    enabled = True
    DIRECTION_CHANNELS = ["POS_X", "NEG_X", "POS_Y", "NEG_Y"]

    def __init__(self, cfg: DatasetCollectorConfig):
        self.cfg = cfg
        self.logger = logger.bind(module_name="DatasetCollector")
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.sample_interval <= 0:
            raise ValueError("dataset_collector.sample_interval must be positive")
        if cfg.shard_size <= 0:
            raise ValueError("dataset_collector.shard_size must be positive")
        if cfg.env_id != 0:
            raise ValueError("dataset_collector.env_id must be 0 for the current collection design")
        if cfg.dtype not in ("float32", "float16"):
            raise ValueError("dataset_collector.dtype must be 'float32' or 'float16'")

        self.storage_dtype = torch.float32 if cfg.dtype == "float32" else torch.float16
        self.buffer: List[Dict[str, torch.Tensor]] = []
        self.shard_idx = 0
        self.total_samples = 0

    def validate(self, runner: Any) -> None:
        if runner.args_cli.my_exp_name != "demo_plane_7":
            raise ValueError(
                "Dataset collection is configured for demo_plane_7. "
                f"Got my_exp_name={runner.args_cli.my_exp_name!r}."
            )
        if runner.env.num_envs != 1:
            raise ValueError(
                "Dataset collection currently supports num_envs=1 and env_id=0 only. "
                f"Got num_envs={runner.env.num_envs}."
            )
        if not isinstance(runner.neural_map, DirectionalBayesianNeuralMap):
            raise TypeError(
                "Dataset collection requires DirectionalBayesianNeuralMap or a subclass. "
                f"Got {type(runner.neural_map).__name__}."
            )
        if not hasattr(runner.neural_map, "maps") or not hasattr(runner.neural_map, "uncertainties"):
            raise TypeError("Neural map must expose 'maps' and 'uncertainties' tensors")
        if runner.neural_map.maps.dim() != 5 or runner.neural_map.maps.shape[3] != 4:
            raise ValueError(
                "Directional neural map must have shape [num_envs, x, y, 4, latent_dim]. "
                f"Got {tuple(runner.neural_map.maps.shape)}."
            )
        if runner.neural_map.uncertainties.dim() != 4 or runner.neural_map.uncertainties.shape[3] != 4:
            raise ValueError(
                "Neural map uncertainty must have shape [num_envs, x, y, 4]. "
                f"Got {tuple(runner.neural_map.uncertainties.shape)}."
            )
        if runner.cost_map.maps.dim() != 4:
            raise ValueError(
                "Cost map must have shape [num_envs, x, y, channels]. "
                f"Got {tuple(runner.cost_map.maps.shape)}."
            )

        self._write_metadata(runner)
        self.logger.info("Dataset collector initialized at {}", self.output_dir)

    def collect(
        self,
        *,
        step: int,
        neural_map: Any,
        cost_map: Any,
        env: Any,
        cost: torch.Tensor,
        dones: torch.Tensor,
        vel_command_generator_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if step % self.cfg.sample_interval != 0:
            return

        env_id = self.cfg.env_id
        sample = {
            "neural_map": self._copy_float_tensor(neural_map.maps[env_id]),
            "neural_map_uncertainty": self._copy_float_tensor(neural_map.uncertainties[env_id]),
            "cost_map": self._copy_float_tensor(cost_map.maps[env_id]),
            "step": torch.tensor(int(step), dtype=torch.long),
            "local_pos": self._copy_float_tensor(env.root_local_pos[env_id, :2]),
            "linear_command": self._copy_float_tensor(env.commands[env_id, :2]),
            "heading_command": self._copy_float_tensor(env.commands[env_id, 3]),
            "root_velocity": self._copy_float_tensor(env.root_states[env_id, 7:9]),
            "current_cost": self._copy_float_tensor(cost[env_id]),
            "done": dones[env_id].detach().cpu().to(dtype=torch.bool),
        }
        self.buffer.append(sample)
        self.total_samples += 1

        if len(self.buffer) >= self.cfg.shard_size:
            self.flush()

    def flush(self) -> None:
        if len(self.buffer) == 0:
            return

        shard = self._stack_buffer()
        shard_path = self.output_dir / f"shard_{self.shard_idx:06d}.pt"
        torch.save(shard, shard_path)
        self.logger.info("Saved dataset shard {} with {} samples", shard_path, len(self.buffer))

        self.buffer.clear()
        self.shard_idx += 1

    def _copy_float_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu().to(dtype=self.storage_dtype).clone()

    def _stack_buffer(self) -> Dict[str, torch.Tensor]:
        keys = self.buffer[0].keys()
        return {key: torch.stack([sample[key] for sample in self.buffer], dim=0) for key in keys}

    def _write_metadata(self, runner: Any) -> None:
        metadata = {
            "env_id": self.cfg.env_id,
            "experiment": runner.args_cli.my_exp_name,
            "sample_interval": self.cfg.sample_interval,
            "shard_size": self.cfg.shard_size,
            "dtype": self.cfg.dtype,
            "direction_channels": self.DIRECTION_CHANNELS,
            "neural_map_type": type(runner.neural_map).__name__,
            "cost_map_type": type(runner.cost_map).__name__,
            "x_range": list(runner.neural_map.cfg.x_range),
            "y_range": list(runner.neural_map.cfg.y_range),
            "least_count": float(runner.neural_map.cfg.least_count),
            "long_latent_dim": int(runner.neural_map.cfg.long_latent_dim),
            "short_latent_dim": int(runner.neural_map.cfg.short_latent_dim),
            "latent_dim": int(runner.neural_map.cfg.long_latent_dim + runner.neural_map.cfg.short_latent_dim),
            "sample_format": {
                "neural_map": "[N, x, y, 4, latent_dim]",
                "neural_map_uncertainty": "[N, x, y, 4]",
                "cost_map": "[N, x, y, cost_channels]",
                "step": "[N]",
                "local_pos": "[N, 2]",
                "linear_command": "[N, 2]",
                "heading_command": "[N]",
                "root_velocity": "[N, 2]",
                "current_cost": "[N]",
                "done": "[N]",
            },
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


def build_dataset_collector(cfg: DatasetCollectorConfig):
    if cfg is None or not cfg.enabled:
        return NullDatasetCollector()
    if cfg.name in ("snapshot", "dataset_collector"):
        return DatasetCollector(cfg)
    if cfg.name in ("real_path", "real_path_dataset_collector"):
        return RealPathDatasetCollector(cfg)
    raise ValueError(f"Unknown dataset collector strategy: {cfg.name!r}")


class RealPathDatasetCollector(DatasetCollector):
    """Collect augmented-style samples from the actually executed rollout path."""

    enabled = True

    def __init__(self, cfg: DatasetCollectorConfig):
        super().__init__(cfg)
        if cfg.velocity_horizon <= 0:
            raise ValueError("dataset_collector.velocity_horizon must be positive")
        if cfg.velocity_command_dt <= 0.0:
            raise ValueError("dataset_collector.velocity_command_dt must be positive")
        if cfg.main_loop_hz <= 0.0:
            raise ValueError("dataset_collector.main_loop_hz must be positive")
        if cfg.max_path_length <= 0:
            raise ValueError("dataset_collector.max_path_length must be positive")
        if cfg.max_waypoints <= 0:
            raise ValueError("dataset_collector.max_waypoints must be positive")
        if cfg.crop_size <= 0:
            raise ValueError("dataset_collector.crop_size must be positive")
        if cfg.uncertainty_log_scale <= 0.0:
            raise ValueError("dataset_collector.uncertainty_log_scale must be positive")
        if cfg.uncertainty_max <= 0.0:
            raise ValueError("dataset_collector.uncertainty_max must be positive")

        self.future_step_interval = max(1, int(round(float(cfg.velocity_command_dt) * float(cfg.main_loop_hz))))
        self.required_future_span = (int(cfg.velocity_horizon) - 1) * self.future_step_interval
        self.num_envs = 0
        self.frames_by_env: List[List[Dict[str, Any]]] = []
        self.frame_idx_by_env: List[int] = []

    def validate(self, runner: Any) -> None:
        if not hasattr(runner.env, "base_lin_vel"):
            raise TypeError("Real path dataset collection requires env.base_lin_vel for local-frame velocity")
        if not hasattr(runner.env, "root_states"):
            raise TypeError("Real path dataset collection requires env.root_states for global heading")
        if not isinstance(runner.neural_map, DirectionalBayesianNeuralMap):
            raise TypeError(
                "Real path dataset collection requires DirectionalBayesianNeuralMap or a subclass. "
                f"Got {type(runner.neural_map).__name__}."
            )
        if not hasattr(runner.neural_map, "maps") or not hasattr(runner.neural_map, "uncertainties"):
            raise TypeError("Neural map must expose 'maps' and 'uncertainties' tensors")
        if runner.neural_map.maps.dim() != 5 or runner.neural_map.maps.shape[3] != 4:
            raise ValueError(
                "Directional neural map must have shape [num_envs, x, y, 4, latent_dim]. "
                f"Got {tuple(runner.neural_map.maps.shape)}."
            )
        if runner.neural_map.uncertainties.dim() != 4 or runner.neural_map.uncertainties.shape[3] != 4:
            raise ValueError(
                "Neural map uncertainty must have shape [num_envs, x, y, 4]. "
                f"Got {tuple(runner.neural_map.uncertainties.shape)}."
            )
        if runner.cost_map.maps.dim() != 4:
            raise ValueError(
                "Cost map must have shape [num_envs, x, y, channels]. "
                f"Got {tuple(runner.cost_map.maps.shape)}."
            )
        if runner.neural_map.maps.shape[0] != runner.env.num_envs:
            raise ValueError(
                "Neural map num_envs must match env.num_envs. "
                f"Got neural_map={runner.neural_map.maps.shape[0]}, env={runner.env.num_envs}."
            )
        if runner.cost_map.maps.shape[0] != runner.env.num_envs:
            raise ValueError(
                "Cost map num_envs must match env.num_envs. "
                f"Got cost_map={runner.cost_map.maps.shape[0]}, env={runner.env.num_envs}."
            )
        self.num_envs = int(runner.env.num_envs)
        self.frames_by_env = [[] for _ in range(self.num_envs)]
        self.frame_idx_by_env = [0 for _ in range(self.num_envs)]
        self._write_metadata(runner)
        self.logger.info("Real path dataset collector initialized at {}", self.output_dir)

    def collect(
        self,
        *,
        step: int,
        neural_map: Any,
        cost_map: Any,
        env: Any,
        cost: torch.Tensor,
        dones: torch.Tensor,
        vel_command_generator_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.frames_by_env:
            self.num_envs = int(env.num_envs)
            self.frames_by_env = [[] for _ in range(self.num_envs)]
            self.frame_idx_by_env = [0 for _ in range(self.num_envs)]

        for env_id in range(int(env.num_envs)):
            env_info = self._get_info_for_env(vel_command_generator_info, env_id)
            reached_goal = bool(env_info.get("reached_goal_flag", False)) if env_info else False
            if bool(dones[env_id].detach().cpu().item()) or reached_goal:
                self._emit_available_samples(env_id, force=True)
                self.frames_by_env[env_id].clear()
                self.frame_idx_by_env[env_id] += 1
                continue

            self.frames_by_env[env_id].append(
                self._capture_frame(step, env_id, neural_map, cost_map, env, cost, dones, vel_command_generator_info)
            )
            self.frame_idx_by_env[env_id] += 1
            self._emit_available_samples(env_id, force=False)

    def flush(self) -> None:
        for env_id in range(len(self.frames_by_env)):
            self._emit_available_samples(env_id, force=True)
            self.frames_by_env[env_id].clear()
        super().flush()

    def _capture_frame(
        self,
        step: int,
        env_id: int,
        neural_map: Any,
        cost_map: Any,
        env: Any,
        cost: torch.Tensor,
        dones: torch.Tensor,
        vel_command_generator_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        start_xy = self._copy_float_tensor(env.root_local_pos[env_id, :2])
        start_grid = self._xy_to_grid(start_xy.float())
        context_goal_xy = self._extract_context_goal_xy(vel_command_generator_info, env_id)
        map_crop, uncertainty_crop, current_latent = self._build_cropped_map_features(
            neural_map.maps[env_id],
            neural_map.uncertainties[env_id],
            start_grid,
        )
        cost_map_crop = self._build_cost_map_crop(cost_map.maps[env_id], start_grid)
        return {
            "_collector_frame_idx": torch.tensor(int(self.frame_idx_by_env[env_id]), dtype=torch.long),
            "_context_goal_xy": context_goal_xy,
            "map_crop": map_crop,
            "uncertainty_crop": uncertainty_crop,
            "cost_map_crop": cost_map_crop,
            "current_latent": current_latent,
            "env_id": torch.tensor(int(env_id), dtype=torch.long),
            "start_grid": torch.tensor(start_grid, dtype=torch.long),
            "start_xy": start_xy,
            "step": torch.tensor(int(step), dtype=torch.long),
            "local_pos": start_xy,
            "linear_command": self._copy_float_tensor(env.commands[env_id, :2]),
            "heading_command": self._copy_float_tensor(env.commands[env_id, 3]),
            "root_velocity": self._copy_float_tensor(env.root_states[env_id, 7:9]),
            "state_local_velocity": self._copy_float_tensor(env.base_lin_vel[env_id, :2]),
            "state_global_heading": self._copy_float_tensor(_yaw_from_quat_xyzw(env.root_states[env_id, 3:7])),
            "current_cost": self._copy_float_tensor(cost[env_id]),
            "done": dones[env_id].detach().cpu().to(dtype=torch.bool),
        }

    def _emit_available_samples(self, env_id: int, *, force: bool) -> None:
        frames = self.frames_by_env[env_id]
        while frames:
            anchor = frames[0]
            if int(anchor["step"].item()) % self.cfg.sample_interval != 0:
                frames.pop(0)
                continue

            available_span = int(frames[-1]["_collector_frame_idx"].item() - anchor["_collector_frame_idx"].item())
            if not force and available_span < self.required_future_span:
                break

            sample = self._build_sample_from_anchor(frames)
            has_full_horizon = bool(sample["velocity_command_mask"].all().item())
            if has_full_horizon or not self.cfg.drop_incomplete_horizon:
                self.buffer.append(sample)
                self.total_samples += 1
                if len(self.buffer) >= self.cfg.shard_size:
                    super().flush()

            frames.pop(0)
            if not force and frames:
                next_anchor_span = int(
                    frames[-1]["_collector_frame_idx"].item() - frames[0]["_collector_frame_idx"].item()
                )
                if next_anchor_span < self.required_future_span:
                    break

    def _build_sample_from_anchor(self, frames: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        anchor = frames[0]
        sequence_frames = self._sample_sequence_frames(frames)
        valid_count = len(sequence_frames)
        velocity_commands = torch.zeros((int(self.cfg.velocity_horizon), 3), dtype=self.storage_dtype)
        velocity_command_mask = torch.zeros((int(self.cfg.velocity_horizon),), dtype=torch.bool)
        for idx, frame in enumerate(sequence_frames):
            velocity_commands[idx, 0:2] = frame["state_local_velocity"].to(dtype=self.storage_dtype)
            velocity_commands[idx, 2] = frame["state_global_heading"].to(dtype=self.storage_dtype)
            velocity_command_mask[idx] = True

        xy_points = [frame["local_pos"].float() for frame in sequence_frames]
        start_xy = anchor["local_pos"].float()
        goal_xy, goal_from_context = self._resolve_goal_xy(anchor, xy_points, start_xy)
        grid_points = [self._xy_to_grid(point) for point in xy_points]
        start_grid = tuple(int(v) for v in anchor["start_grid"].tolist())
        goal_grid = self._xy_to_grid(goal_xy)
        astar_path, astar_path_mask = _pad_grid_points(grid_points, int(self.cfg.max_path_length))
        waypoints_xy, waypoints_mask = _pad_xy_points(xy_points, int(self.cfg.max_waypoints))

        sample = {key: value.detach().cpu().clone() for key, value in anchor.items() if not key.startswith("_")}
        sample.update({
            "start_grid": torch.tensor(start_grid, dtype=torch.long),
            "goal_grid": torch.tensor(goal_grid, dtype=torch.long),
            "goal_xy": goal_xy.detach().cpu().to(dtype=self.storage_dtype).clone(),
            "goal_from_context": torch.tensor(goal_from_context, dtype=torch.bool),
            "astar_path": astar_path,
            "astar_path_mask": astar_path_mask,
            "waypoints_xy": waypoints_xy.to(dtype=self.storage_dtype),
            "waypoints_mask": waypoints_mask,
            "initial_heading": anchor["state_global_heading"].detach().cpu().to(dtype=self.storage_dtype).clone(),
            "velocity_commands": velocity_commands,
            "velocity_command_mask": velocity_command_mask,
            "plan_total_cost": self._sum_sequence_cost(sequence_frames[:valid_count]),
        })
        return sample

    def _extract_context_goal_xy(self, info: Optional[Dict[str, Any]], env_id: int) -> Optional[torch.Tensor]:
        if not info:
            return None
        env_info = self._get_info_for_env(info, env_id)
        if env_info is not info:
            env_goal_xy = self._extract_context_goal_xy(env_info, env_id)
            if env_goal_xy is not None:
                return env_goal_xy
        if "goal_xy" not in info:
            return None
        goal_xy = info["goal_xy"]
        if goal_xy is None:
            return None
        goal_tensor = torch.as_tensor(goal_xy, dtype=torch.float32)
        if goal_tensor.numel() < 2:
            return None
        return goal_tensor.reshape(-1)[:2].detach().cpu().to(dtype=self.storage_dtype).clone()

    def _get_info_for_env(self, info: Optional[Dict[str, Any]], env_id: int) -> Optional[Dict[str, Any]]:
        if not info:
            return None
        per_env_info = info.get("per_env")
        if isinstance(per_env_info, Sequence) and env_id < len(per_env_info):
            env_info = per_env_info[env_id]
            if isinstance(env_info, dict):
                return env_info
        return info

    def _resolve_goal_xy(
        self,
        anchor: Dict[str, Any],
        xy_points: Sequence[torch.Tensor],
        start_xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        context_goal_xy = anchor.get("_context_goal_xy")
        if context_goal_xy is not None:
            return context_goal_xy.float(), True
        return (xy_points[-1] if xy_points else start_xy), False

    def _build_cropped_map_features(
        self,
        neural_map: torch.Tensor,
        uncertainty: torch.Tensor,
        start_grid: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_grid_tensor = torch.tensor(start_grid, dtype=torch.long, device=neural_map.device)
        map_crop = crop_directional_map(
            neural_map.float(),
            start_grid_tensor,
            crop_size=int(self.cfg.crop_size),
            pad_value=0.0,
        )
        uncertainty_crop = crop_uncertainty_map(
            uncertainty.float(),
            start_grid_tensor,
            crop_size=int(self.cfg.crop_size),
            pad_value=float(self.cfg.uncertainty_max),
        )
        uncertainty_crop = scale_uncertainty_crop(
            uncertainty_crop,
            uncertainty_log_scale=float(self.cfg.uncertainty_log_scale),
        )
        current_latent = gather_directional_latent(neural_map.float(), start_grid_tensor)
        return (
            map_crop.detach().cpu().to(dtype=self.storage_dtype).clone(),
            uncertainty_crop.detach().cpu().to(dtype=self.storage_dtype).clone(),
            current_latent.detach().cpu().to(dtype=self.storage_dtype).clone(),
        )

    def _build_cost_map_crop(
        self,
        cost_map: torch.Tensor,
        start_grid: Tuple[int, int],
    ) -> torch.Tensor:
        start_grid_tensor = torch.tensor(start_grid, dtype=torch.long, device=cost_map.device)
        cost_crop = crop_cost_map(
            cost_map.float(),
            start_grid_tensor,
            crop_size=int(self.cfg.crop_size),
            pad_value=0.0,
        )
        return cost_crop.detach().cpu().to(dtype=self.storage_dtype).clone()

    def _sum_sequence_cost(self, sequence_frames: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if len(sequence_frames) == 0:
            return torch.tensor(0.0, dtype=self.storage_dtype)
        return torch.stack([frame["current_cost"].float() for frame in sequence_frames], dim=0).sum().to(
            dtype=self.storage_dtype
        )

    def _sample_sequence_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        selected = []
        anchor_frame_idx = int(frames[0]["_collector_frame_idx"].item())
        frame_by_step = {int(frame["_collector_frame_idx"].item()): frame for frame in frames}
        for idx in range(int(self.cfg.velocity_horizon)):
            target_step = anchor_frame_idx + idx * self.future_step_interval
            frame = frame_by_step.get(target_step)
            if frame is None:
                break
            selected.append(frame)
        return selected

    def _xy_to_grid(self, xy: torch.Tensor) -> Tuple[int, int]:
        x_idx = int(torch.floor((xy[0].float() - float(self._x_range[0])) / float(self._least_count)).item())
        y_idx = int(torch.floor((xy[1].float() - float(self._y_range[0])) / float(self._least_count)).item())
        h, w = self._map_shape
        return (max(0, min(x_idx, h - 1)), max(0, min(y_idx, w - 1)))

    def _write_metadata(self, runner: Any) -> None:
        self._x_range = list(runner.neural_map.cfg.x_range)
        self._y_range = list(runner.neural_map.cfg.y_range)
        self._least_count = float(runner.neural_map.cfg.least_count)
        self._map_shape = tuple(int(v) for v in runner.neural_map.maps.shape[1:3])
        super()._write_metadata(runner)
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata["schema_version"] = "cropped_v1"
        metadata["crop_size"] = int(self.cfg.crop_size)
        metadata["uncertainty_log_scale"] = float(self.cfg.uncertainty_log_scale)
        metadata["uncertainty_max"] = float(self.cfg.uncertainty_max)
        metadata["collector"] = {
            "name": self.cfg.name,
            "schema_version": "cropped_v1",
            "num_envs": int(getattr(self, "num_envs", 0)),
            "crop_size": int(self.cfg.crop_size),
            "uncertainty_log_scale": float(self.cfg.uncertainty_log_scale),
            "uncertainty_max": float(self.cfg.uncertainty_max),
            "velocity_horizon": int(self.cfg.velocity_horizon),
            "velocity_command_dt": float(self.cfg.velocity_command_dt),
            "cost_obstacle_threshold": float(self.cfg.cost_obstacle_threshold),
            "robot_radius_cells": int(self.cfg.robot_radius_cells),
            "main_loop_hz": float(self.cfg.main_loop_hz),
            "future_step_interval": int(self.future_step_interval),
            "drop_incomplete_horizon": bool(self.cfg.drop_incomplete_horizon),
            "velocity_source": "env.base_lin_vel[:, :2]",
            "heading_source": "yaw(env.root_states[:, 3:7])",
            "goal_source": "vel_command_generator_info.goal_xy with sequence-end fallback",
        }
        metadata["sample_format"] = {
            "map_crop": "[N, 4 * latent_dim, crop_size, crop_size]",
            "uncertainty_crop": "[N, 4, crop_size, crop_size]",
            "cost_map_crop": "[N, cost_channels, crop_size, crop_size]",
            "current_latent": "[N, 4 * latent_dim]",
            "env_id": "[N]",
            "start_grid": "[N, 2]",
            "goal_grid": "[N, 2]",
            "start_xy": "[N, 2]",
            "goal_xy": "[N, 2]",
            "goal_from_context": "[N]",
            "astar_path": "[N, max_path_length, 2]",
            "astar_path_mask": "[N, max_path_length]",
            "waypoints_xy": "[N, max_waypoints, 2]",
            "waypoints_mask": "[N, max_waypoints]",
            "initial_heading": "[N]",
            "velocity_commands": "[N, velocity_horizon, 3]",
            "velocity_command_mask": "[N, velocity_horizon]",
            "plan_total_cost": "[N]",
            "step": "[N]",
            "local_pos": "[N, 2]",
            "linear_command": "[N, 2]",
            "heading_command": "[N]",
            "root_velocity": "[N, 2]",
            "state_local_velocity": "[N, 2]",
            "state_global_heading": "[N]",
            "current_cost": "[N]",
            "done": "[N]",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def _yaw_from_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _pad_grid_points(points: Sequence[Tuple[int, int]], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    padded = torch.full((max_length, 2), -1, dtype=torch.long)
    mask = torch.zeros((max_length,), dtype=torch.bool)
    clipped = list(points[:max_length])
    if clipped:
        padded[:len(clipped)] = torch.tensor(clipped, dtype=torch.long)
        mask[:len(clipped)] = True
    return padded, mask


def _pad_xy_points(points: Sequence[torch.Tensor], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    padded = torch.zeros((max_length, 2), dtype=torch.float32)
    mask = torch.zeros((max_length,), dtype=torch.bool)
    clipped = list(points[:max_length])
    if clipped:
        padded[:len(clipped)] = torch.stack([point.detach().cpu().float() for point in clipped], dim=0)
        mask[:len(clipped)] = True
    return padded, mask
