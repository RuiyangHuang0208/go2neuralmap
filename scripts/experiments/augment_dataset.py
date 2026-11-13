#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from tqdm import tqdm
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import isaacgym  # noqa: F401
import torch
from loguru import logger
from omegaconf import OmegaConf

from utils.diffusion_policy.dataset import (
    crop_cost_map,
    crop_directional_map,
    crop_uncertainty_map,
    gather_directional_latent,
    scale_uncertainty_crop,
)
from utils.navigation import PlannerState
from utils.navigation.registry import PlannerRegistry
from utils.position_controller import (
    build_refined_waypoints_from_plan,
    compute_bezier_velocity_command_sequence,
    grid_to_xy,
)
from utils.position_controller.registry import PositionControllerRegistry


GridPoint = Tuple[int, int]


@dataclass
class AugmentationStats:
    raw_samples: int = 0
    attempts: int = 0
    successes: int = 0
    sampling_failures: int = 0
    planning_failures: int = 0
    truncated_paths: int = 0
    truncated_waypoints: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment cost-map dataset with A* plans and local Bezier commands.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def load_configs(args: argparse.Namespace):
    hydra.initialize(config_path="configs", version_base="1.1")
    cfg_omega = hydra.compose(config_name="data_augmentation", overrides=args.overrides)
    OmegaConf.resolve(cfg_omega)
    return cfg_omega


def merge_config(cfg_obj, cfg_dict):
    schema = OmegaConf.structured(cfg_obj)
    merged_cfg = OmegaConf.merge(schema, cfg_dict)
    return OmegaConf.to_object(merged_cfg)


def build_path_planner(module_cfg_omega):
    cls_, cfg_obj = PlannerRegistry.get(module_cfg_omega.name)
    if cls_ is None:
        raise ValueError(f"Unknown planner: {module_cfg_omega.name}")
    if module_cfg_omega.config is not None:
        cfg_obj = merge_config(cfg_obj, module_cfg_omega.config)
    return cls_(cfg_obj)


def build_position_controller(module_cfg_omega):
    cls_, cfg_obj = PositionControllerRegistry.get(module_cfg_omega.name)
    if cls_ is None:
        raise ValueError(f"Unknown position controller: {module_cfg_omega.name}")
    if module_cfg_omega.config is not None:
        cfg_obj = merge_config(cfg_obj, module_cfg_omega.config)
    return cls_(cfg_obj)


def iter_shard_samples(raw_dataset: Path) -> Iterable[Dict[str, torch.Tensor]]:
    shard_paths = sorted(raw_dataset.glob("shard_*.pt"))
    if len(shard_paths) == 0:
        raise FileNotFoundError(f"No shard_*.pt files found in {raw_dataset}")

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        first_key = next(iter(shard.keys()))
        count = int(shard[first_key].shape[0])
        for idx in range(count):
            yield {key: value[idx] for key, value in shard.items()}


def load_metadata(raw_dataset: Path) -> Dict[str, Any]:
    metadata_path = raw_dataset / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {raw_dataset}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def valid_grid_points(cost_map: torch.Tensor, obstacle_threshold: float) -> torch.Tensor:
    if cost_map.dim() != 3 or cost_map.shape[2] != 4:
        raise ValueError(f"Expected directional cost_map with shape [H, W, 4], got {tuple(cost_map.shape)}")
    finite_mask = torch.isfinite(cost_map).all(dim=-1)
    traversable_mask = (cost_map < float(obstacle_threshold)).any(dim=-1)
    return torch.nonzero(finite_mask & traversable_mask, as_tuple=False)


def sample_start_goal(
    valid_points: torch.Tensor,
    *,
    least_count: float,
    min_dist: float,
    max_dist: Optional[float],
    max_tries: int,
    rng: random.Random,
) -> Optional[Tuple[GridPoint, GridPoint]]:
    if valid_points.shape[0] < 2:
        return None

    num_points = int(valid_points.shape[0])
    for _ in range(max_tries):
        start_idx = rng.randrange(num_points)
        goal_idx = rng.randrange(num_points)
        if start_idx == goal_idx:
            continue

        start = tuple(int(v) for v in valid_points[start_idx].tolist())
        goal = tuple(int(v) for v in valid_points[goal_idx].tolist())
        dx = (goal[0] - start[0]) * float(least_count)
        dy = (goal[1] - start[1]) * float(least_count)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            continue
        if max_dist is not None and dist > max_dist:
            continue
        return start, goal

    return None


def pad_grid_path(path: List[GridPoint], max_length: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    truncated = len(path) > max_length
    clipped = path[:max_length]
    padded = torch.full((max_length, 2), -1, dtype=torch.long)
    mask = torch.zeros((max_length,), dtype=torch.bool)
    if len(clipped) > 0:
        padded[:len(clipped)] = torch.tensor(clipped, dtype=torch.long)
        mask[:len(clipped)] = True
    return padded, mask, truncated


def pad_xy_points(points: List[torch.Tensor], max_length: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    truncated = len(points) > max_length
    clipped = points[:max_length]
    padded = torch.zeros((max_length, 2), dtype=torch.float32)
    mask = torch.zeros((max_length,), dtype=torch.bool)
    if len(clipped) > 0:
        padded[:len(clipped)] = torch.stack([point.detach().cpu().float() for point in clipped], dim=0)
        mask[:len(clipped)] = True
    return padded, mask, truncated


def stack_and_save(buffer: List[Dict[str, torch.Tensor]], output_dataset: Path, shard_idx: int) -> int:
    if len(buffer) == 0:
        return shard_idx
    output_dataset.mkdir(parents=True, exist_ok=True)
    keys = buffer[0].keys()
    shard = {key: torch.stack([sample[key] for sample in buffer], dim=0) for key in keys}
    shard_path = output_dataset / f"shard_{shard_idx:06d}.pt"
    torch.save(shard, shard_path)
    logger.info("Saved {} samples to {}", len(buffer), shard_path)
    buffer.clear()
    return shard_idx + 1


def storage_dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported output dtype: {dtype_name!r}")


def build_cropped_features(
    sample: Dict[str, torch.Tensor],
    start_grid: GridPoint,
    *,
    crop_size: int,
    uncertainty_max: float,
    uncertainty_log_scale: float,
    storage_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_grid_tensor = torch.tensor(start_grid, dtype=torch.long)
    neural_map = sample["neural_map"].detach().cpu().float()
    uncertainty = sample["neural_map_uncertainty"].detach().cpu().float()
    map_crop = crop_directional_map(
        neural_map,
        start_grid_tensor,
        crop_size=int(crop_size),
        pad_value=0.0,
    )
    uncertainty_crop = crop_uncertainty_map(
        uncertainty,
        start_grid_tensor,
        crop_size=int(crop_size),
        pad_value=float(uncertainty_max),
    )
    uncertainty_crop = scale_uncertainty_crop(
        uncertainty_crop,
        uncertainty_log_scale=float(uncertainty_log_scale),
    )
    current_latent = gather_directional_latent(neural_map, start_grid_tensor)
    return (
        map_crop.to(dtype=storage_dtype).clone(),
        uncertainty_crop.to(dtype=storage_dtype).clone(),
        current_latent.to(dtype=storage_dtype).clone(),
    )


def build_cost_map_crop(
    cost_map: torch.Tensor,
    start_grid: GridPoint,
    *,
    crop_size: int,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    start_grid_tensor = torch.tensor(start_grid, dtype=torch.long)
    cost_crop = crop_cost_map(
        cost_map.float(),
        start_grid_tensor,
        crop_size=int(crop_size),
        pad_value=0.0,
    )
    return cost_crop.detach().cpu().to(dtype=storage_dtype).clone()


def augment_dataset(cfg_omega) -> AugmentationStats:
    aug_cfg = cfg_omega.augmentation
    raw_dataset = Path(aug_cfg.raw_dataset).expanduser()
    output_dataset = Path(aug_cfg.output_dataset).expanduser()
    if not raw_dataset.exists():
        raise FileNotFoundError(f"raw_dataset does not exist: {raw_dataset}")
    if str(output_dataset) == "":
        raise ValueError("augmentation.output_dataset must be set")

    metadata = load_metadata(raw_dataset)
    x_range = metadata["x_range"]
    y_range = metadata["y_range"]
    least_count = float(metadata["least_count"])

    planner = build_path_planner(cfg_omega.modules.planner)
    position_controller = build_position_controller(cfg_omega.modules.position_controller)
    rng = random.Random(int(aug_cfg.seed))
    device = torch.device("cpu")
    stats = AugmentationStats()
    crop_size = int(OmegaConf.select(aug_cfg, "crop_size", default=8))
    uncertainty_log_scale = float(OmegaConf.select(aug_cfg, "uncertainty_log_scale", default=12.0))
    uncertainty_max = float(OmegaConf.select(aug_cfg, "uncertainty_max", default=100000.0))
    output_dtype = str(OmegaConf.select(aug_cfg, "output_dtype", default="float16"))
    storage_dtype = storage_dtype_from_name(output_dtype)

    output_dataset.mkdir(parents=True, exist_ok=True)
    output_metadata = dict(metadata)
    output_metadata["schema_version"] = "cropped_v1"
    output_metadata["dtype"] = output_dtype
    output_metadata["crop_size"] = crop_size
    output_metadata["uncertainty_log_scale"] = uncertainty_log_scale
    output_metadata["uncertainty_max"] = uncertainty_max
    output_metadata["augmentation"] = OmegaConf.to_container(aug_cfg, resolve=True)
    output_metadata["planner"] = OmegaConf.to_container(cfg_omega.modules.planner, resolve=True)
    output_metadata["position_controller"] = OmegaConf.to_container(cfg_omega.modules.position_controller, resolve=True)
    output_metadata["sample_format"] = {
        "map_crop": "[N, 4 * latent_dim, crop_size, crop_size]",
        "uncertainty_crop": "[N, 4, crop_size, crop_size]",
        "cost_map_crop": "[N, cost_channels, crop_size, crop_size]",
        "current_latent": "[N, 4 * latent_dim]",
        "start_grid": "[N, 2]",
        "goal_grid": "[N, 2]",
        "start_xy": "[N, 2]",
        "goal_xy": "[N, 2]",
        "astar_path": "[N, max_path_length, 2]",
        "astar_path_mask": "[N, max_path_length]",
        "waypoints_xy": "[N, max_waypoints, 2]",
        "waypoints_mask": "[N, max_waypoints]",
        "initial_heading": "[N]",
        "velocity_commands": "[N, velocity_horizon, 3]",
        "velocity_command_mask": "[N, velocity_horizon]",
        "plan_total_cost": "[N]",
    }
    with open(output_dataset / "metadata.json", "w") as f:
        json.dump(output_metadata, f, indent=2)

    buffer: List[Dict[str, torch.Tensor]] = []
    shard_idx = 0
    max_raw_samples = aug_cfg.max_raw_samples
    num_aug_per_sample = int(aug_cfg.num_aug_per_sample)

    for sample_idx, sample in tqdm(enumerate(iter_shard_samples(raw_dataset))):
        if max_raw_samples is not None and sample_idx >= int(max_raw_samples):
            break

        stats.raw_samples += 1
        cost_map = sample["cost_map"].detach().cpu().float()
        valid_points = valid_grid_points(cost_map, float(planner.cfg.obstacle_threshold))

        for _ in range(num_aug_per_sample):
            stats.attempts += 1
            pair = sample_start_goal(
                valid_points,
                least_count=least_count,
                min_dist=float(aug_cfg.min_start_goal_dist),
                max_dist=None if aug_cfg.max_start_goal_dist is None else float(aug_cfg.max_start_goal_dist),
                max_tries=int(aug_cfg.max_sampling_tries),
                rng=rng,
            )
            if pair is None:
                stats.sampling_failures += 1
                continue

            start_grid, goal_grid = pair
            result = planner.plan(start_grid, goal_grid, cost_map)
            if result.state != PlannerState.SUCCESS or result.path is None or len(result.path) < 2:
                stats.planning_failures += 1
                continue

            path = result.path
            waypoint_indices = planner.subsample_waypoints(result)
            goal_xy = grid_to_xy(goal_grid, x_range, y_range, least_count, device)
            waypoints = build_refined_waypoints_from_plan(
                path=path,
                waypoint_indices=waypoint_indices,
                goal_xy=goal_xy,
                x_range=x_range,
                y_range=y_range,
                least_count=least_count,
                device=device,
            )
            velocity_commands, velocity_mask, initial_heading = compute_bezier_velocity_command_sequence(
                position_controller=position_controller,
                path=path,
                waypoint_indices=waypoint_indices,
                waypoints=waypoints,
                x_range=x_range,
                y_range=y_range,
                least_count=least_count,
                horizon=int(aug_cfg.velocity_horizon),
                device=device,
                command_dt=float(aug_cfg.velocity_command_dt),
                initial_heading_method=str(aug_cfg.initial_heading_method),
                heading_rng=rng,
                return_initial_heading=True,
            )

            astar_path, astar_path_mask, path_truncated = pad_grid_path(path, int(aug_cfg.max_path_length))
            waypoints_xy, waypoints_mask, waypoints_truncated = pad_xy_points(waypoints, int(aug_cfg.max_waypoints))
            stats.truncated_paths += int(path_truncated)
            stats.truncated_waypoints += int(waypoints_truncated)

            map_crop, uncertainty_crop, current_latent = build_cropped_features(
                sample,
                start_grid,
                crop_size=crop_size,
                uncertainty_max=uncertainty_max,
                uncertainty_log_scale=uncertainty_log_scale,
                storage_dtype=storage_dtype,
            )
            cost_map_crop = build_cost_map_crop(
                cost_map,
                start_grid,
                crop_size=crop_size,
                storage_dtype=storage_dtype,
            )
            augmented = {
                "map_crop": map_crop,
                "uncertainty_crop": uncertainty_crop,
                "cost_map_crop": cost_map_crop,
                "current_latent": current_latent,
                "start_grid": torch.tensor(start_grid, dtype=torch.long),
                "goal_grid": torch.tensor(goal_grid, dtype=torch.long),
                "start_xy": grid_to_xy(start_grid, x_range, y_range, least_count, device).cpu().to(dtype=storage_dtype),
                "goal_xy": goal_xy.cpu().to(dtype=storage_dtype),
                "astar_path": astar_path,
                "astar_path_mask": astar_path_mask,
                "waypoints_xy": waypoints_xy.to(dtype=storage_dtype),
                "waypoints_mask": waypoints_mask,
                "initial_heading": initial_heading.cpu().to(dtype=storage_dtype),
                "velocity_commands": velocity_commands.cpu().to(dtype=storage_dtype),
                "velocity_command_mask": velocity_mask.cpu(),
                "plan_total_cost": torch.tensor(float(result.total_cost), dtype=storage_dtype),
            }
            buffer.append(augmented)
            stats.successes += 1

            if len(buffer) >= int(aug_cfg.output_shard_size):
                shard_idx = stack_and_save(buffer, output_dataset, shard_idx)

    stack_and_save(buffer, output_dataset, shard_idx)
    return stats


# @hydra.main(config_path="configs", config_name="data_augmentation", version_base="1.1")
def main() -> int:
    args = parse_args()
    cfg_omega = load_configs(args)
    logger.info("Composed config:\n{}", OmegaConf.to_yaml(cfg_omega, resolve=False))
    stats = augment_dataset(cfg_omega)
    logger.info("Augmentation stats: {}", stats)
    if stats.successes == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
