from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tqdm import tqdm


REQUIRED_AUGMENTED_KEYS = (
    "map_crop",
    "uncertainty_crop",
    "current_latent",
    "start_grid",
    "start_xy",
    "goal_xy",
    "initial_heading",
    "velocity_commands",
    "velocity_command_mask",
)

COST_AUGMENTED_KEYS = (
    "cost_map_crop",
)

CURRENT_HEADING_SOURCE_KEYS = {
    "initial_heading": "initial_heading",
    "state_global_heading": "state_global_heading",
}


@dataclass
class TerminalCommandPaddingConfig:
    enabled: bool = False
    near_threshold: float = 1.0
    goal_tolerance: float = 0.35
    only_if_reached_goal: bool = True


@dataclass
class AugmentedNeuralMapDatasetConfig:
    dataset_dir: str = "go2_smoke"
    max_samples: Optional[int] = None
    crop_size: int = 8
    goal_scale: float = 12.0
    velocity_scale: float = 1.5
    uncertainty_log_scale: float = 12.0
    uncertainty_max: float = 100000.0
    shard_cache_size: int = 8
    require_cost_map: bool = False
    current_heading_source: str = "initial_heading"
    terminal_padding: TerminalCommandPaddingConfig = field(default_factory=TerminalCommandPaddingConfig)


class AugmentedNeuralMapDataset(Dataset):
    """Lazy shard dataset for cropped NeuralMap command-sequence labels."""

    def __init__(self, cfg: AugmentedNeuralMapDatasetConfig):
        self.cfg = cfg
        self.dataset_dir = Path(cfg.dataset_dir).expanduser()
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"dataset_dir does not exist: {self.dataset_dir}")
        if cfg.crop_size <= 0:
            raise ValueError("crop_size must be positive")
        if cfg.goal_scale <= 0.0:
            raise ValueError("goal_scale must be positive")
        if cfg.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")
        if cfg.shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        if str(cfg.current_heading_source) not in CURRENT_HEADING_SOURCE_KEYS:
            raise ValueError(
                "current_heading_source must be one of "
                f"{sorted(CURRENT_HEADING_SOURCE_KEYS)}, got {cfg.current_heading_source!r}"
            )
        if cfg.terminal_padding.near_threshold < 0.0:
            raise ValueError("terminal_padding.near_threshold must be non-negative")
        if cfg.terminal_padding.goal_tolerance < 0.0:
            raise ValueError("terminal_padding.goal_tolerance must be non-negative")

        self.metadata = self._load_metadata()
        self.shard_paths = sorted(self.dataset_dir.glob("shard_*.pt"))
        if len(self.shard_paths) == 0:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.dataset_dir}")

        self.index: List[Tuple[int, int]] = []
        self._shard_lengths: List[int] = []
        for shard_idx, shard_path in tqdm(enumerate(self.shard_paths)):
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._validate_shard(shard_path, shard)
            count = int(shard["start_grid"].shape[0])
            self._shard_lengths.append(count)
            for local_idx in range(count):
                if cfg.max_samples is not None and len(self.index) >= int(cfg.max_samples):
                    break
                self.index.append((shard_idx, local_idx))
            del shard
            if cfg.max_samples is not None and len(self.index) >= int(cfg.max_samples):
                break

        if len(self.index) == 0:
            raise ValueError("Dataset index is empty")

        self._cache: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.latent_dim = int(self.metadata.get("latent_dim", 0))
        if self.latent_dim <= 0:
            first = self._load_shard(0)
            self.latent_dim = int(first["current_latent"].shape[-1]) // 4
        self.horizon = int(self._load_shard(0)["velocity_commands"].shape[1])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, local_idx = self.index[int(idx)]
        shard = self._load_shard(shard_idx)
        wanted_keys = set(REQUIRED_AUGMENTED_KEYS)
        wanted_keys.add(CURRENT_HEADING_SOURCE_KEYS[str(self.cfg.current_heading_source)])
        if self.cfg.require_cost_map:
            wanted_keys.update(COST_AUGMENTED_KEYS)
        sample = {key: value[local_idx] for key, value in shard.items() if key in wanted_keys}

        map_crop = sample["map_crop"].float()
        uncertainty_crop = sample["uncertainty_crop"].float()
        current_latent = sample["current_latent"].float()
        goal_delta = (sample["goal_xy"].float() - sample["start_xy"].float()) / float(self.cfg.goal_scale)
        initial_heading = sample["initial_heading"].float()
        heading_key = CURRENT_HEADING_SOURCE_KEYS[str(self.cfg.current_heading_source)]
        current_heading = sample[heading_key].float()
        # current_heading_features = torch.stack([torch.sin(current_heading), torch.cos(current_heading)])
        current_heading_features = torch.stack([torch.zeros_like(current_heading), torch.ones_like(current_heading)])

        velocity_commands = sample["velocity_commands"].float()
        command_mask = sample["velocity_command_mask"].bool()
        velocity_commands, command_mask = self._terminal_pad_commands(
            velocity_commands,
            command_mask,
            start_xy=sample["start_xy"].float(),
            goal_xy=sample["goal_xy"].float(),
        )

        command_target = commands_to_training_target(
            velocity_commands,
            velocity_scale=float(self.cfg.velocity_scale),
        )

        return {
            "map_crop": map_crop,
            "uncertainty_crop": uncertainty_crop,
            "goal_delta": goal_delta,
            "current_latent": current_latent,
            "current_heading": current_heading_features,
            "initial_heading": initial_heading,
            "target": command_target,
            "target_mask": command_mask,
            **self._cost_sample_fields(sample),
        }

    def _load_metadata(self) -> Dict:
        metadata_path = self.dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.dataset_dir}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        schema_version = metadata.get("schema_version")
        if schema_version != "cropped_v1":
            raise ValueError(
                f"Unsupported dataset schema_version={schema_version!r}; expected 'cropped_v1'. "
                "Regenerate the dataset with the cropped schema."
            )
        return metadata

    def _validate_shard(self, shard_path: Path, shard: Dict[str, torch.Tensor]) -> None:
        missing = [key for key in REQUIRED_AUGMENTED_KEYS if key not in shard]
        heading_key = CURRENT_HEADING_SOURCE_KEYS[str(self.cfg.current_heading_source)]
        if heading_key not in shard and heading_key not in missing:
            missing.append(heading_key)
        if self.cfg.require_cost_map:
            missing.extend([key for key in COST_AUGMENTED_KEYS if key not in shard])
        if missing:
            raise ValueError(f"{shard_path} is missing augmented dataset keys: {missing}")
        count = int(shard["start_grid"].shape[0])
        for key in REQUIRED_AUGMENTED_KEYS:
            if int(shard[key].shape[0]) != count:
                raise ValueError(f"{shard_path}:{key} has inconsistent first dimension")
        if shard["map_crop"].dim() != 4:
            raise ValueError(f"{shard_path}:map_crop must have shape [N, C, H, W]")
        if shard["uncertainty_crop"].dim() != 4:
            raise ValueError(f"{shard_path}:uncertainty_crop must have shape [N, 4, H, W]")
        crop_h = int(shard["map_crop"].shape[-2])
        crop_w = int(shard["map_crop"].shape[-1])
        if crop_h != int(self.cfg.crop_size) or crop_w != int(self.cfg.crop_size):
            raise ValueError(f"{shard_path}:map_crop crop size does not match cfg.crop_size={self.cfg.crop_size}")
        if self.cfg.require_cost_map:
            if shard["cost_map_crop"].dim() != 4:
                raise ValueError(f"{shard_path}:cost_map_crop must have shape [N, C, H, W]")
            cost_crop_h = int(shard["cost_map_crop"].shape[-2])
            cost_crop_w = int(shard["cost_map_crop"].shape[-1])
            if cost_crop_h != int(self.cfg.crop_size) or cost_crop_w != int(self.cfg.crop_size):
                raise ValueError(
                    f"{shard_path}:cost_map_crop crop size does not match cfg.crop_size={self.cfg.crop_size}"
                )

    def _load_shard(self, shard_idx: int) -> Dict[str, torch.Tensor]:
        shard_idx = int(shard_idx)
        if shard_idx in self._cache:
            shard = self._cache.pop(shard_idx)
            self._cache[shard_idx] = shard
            return shard

        shard = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
        self._cache[shard_idx] = shard
        while len(self._cache) > int(self.cfg.shard_cache_size):
            self._cache.popitem(last=False)
        return shard

    def _cost_sample_fields(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.cfg.require_cost_map:
            return {}
        metadata = self.metadata
        return {
            "cost_map_crop": sample["cost_map_crop"].float(),
            "least_count": torch.tensor(float(metadata.get("least_count", 1.0)), dtype=torch.float32),
            "velocity_command_dt": torch.tensor(
                float(metadata.get("collector", {}).get("velocity_command_dt", metadata.get("velocity_command_dt", 0.5))),
                dtype=torch.float32,
            ),
        }

    def _terminal_pad_commands(
        self,
        commands: torch.Tensor,
        mask: torch.Tensor,
        *,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.terminal_padding
        if not bool(cfg.enabled) or bool(mask.all()):
            return commands, mask
        valid_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if int(valid_indices.numel()) == 0:
            return commands, mask

        start_goal_dist = torch.norm(goal_xy.float() - start_xy.float())
        if float(start_goal_dist.item()) > float(cfg.near_threshold):
            return commands, mask

        if bool(cfg.only_if_reached_goal):
            path = integrate_velocity_commands_xy(
                start_xy.float(),
                commands[mask].float(),
                dt=self._velocity_command_dt(),
            )
            min_goal_dist = torch.norm(path - goal_xy.float().view(1, 2), dim=-1).min()
            if float(min_goal_dist.item()) > float(cfg.goal_tolerance):
                return commands, mask

        padded_commands = commands.clone()
        padded_mask = mask.clone()
        last_valid_idx = int(valid_indices[-1].item())
        last_heading = padded_commands[last_valid_idx, 2].clone()
        invalid_indices = torch.nonzero(~padded_mask, as_tuple=False).flatten()
        if int(invalid_indices.numel()) == 0:
            return padded_commands, padded_mask
        padded_commands[invalid_indices, 0:2] = 0.0
        padded_commands[invalid_indices, 2] = last_heading
        padded_mask[invalid_indices] = True
        return padded_commands, padded_mask

    def _velocity_command_dt(self) -> float:
        metadata = self.metadata
        return float(metadata.get("collector", {}).get("velocity_command_dt", metadata.get("velocity_command_dt", 0.5)))


def crop_directional_map(
    neural_map: torch.Tensor,
    start_grid: torch.Tensor,
    *,
    crop_size: int,
    pad_value: float,
) -> torch.Tensor:
    crop = _crop_xy_tensor(neural_map, start_grid, crop_size=crop_size, pad_value=pad_value)
    return crop.permute(2, 3, 0, 1).contiguous().view(-1, crop_size, crop_size)


def crop_uncertainty_map(
    uncertainty: torch.Tensor,
    start_grid: torch.Tensor,
    *,
    crop_size: int,
    pad_value: float,
) -> torch.Tensor:
    crop = _crop_xy_tensor(uncertainty, start_grid, crop_size=crop_size, pad_value=pad_value)
    return crop.permute(2, 0, 1).contiguous()


def crop_cost_map(
    cost_map: torch.Tensor,
    start_grid: torch.Tensor,
    *,
    crop_size: int,
    pad_value: float,
) -> torch.Tensor:
    crop = _crop_xy_tensor(cost_map, start_grid, crop_size=crop_size, pad_value=pad_value)
    if crop.dim() == 2:
        return crop.unsqueeze(0).contiguous()
    return crop.permute(2, 0, 1).contiguous()


def scale_uncertainty_crop(
    uncertainty_crop: torch.Tensor,
    *,
    uncertainty_log_scale: float,
) -> torch.Tensor:
    if uncertainty_log_scale <= 0.0:
        raise ValueError("uncertainty_log_scale must be positive")
    scaled = torch.log1p(torch.clamp(uncertainty_crop.float(), min=0.0))
    return torch.clamp(scaled / float(uncertainty_log_scale), min=0.0, max=1.0)


def gather_directional_latent(neural_map: torch.Tensor, start_grid: torch.Tensor) -> torch.Tensor:
    x = int(start_grid[0].item())
    y = int(start_grid[1].item())
    if x < 0 or y < 0 or x >= neural_map.shape[0] or y >= neural_map.shape[1]:
        return neural_map.new_zeros((neural_map.shape[2] * neural_map.shape[3],))
    return neural_map[x, y].contiguous().view(-1)


def infer_initial_heading(astar_path: torch.Tensor, astar_path_mask: torch.Tensor) -> torch.Tensor:
    valid_count = int(astar_path_mask.bool().sum().item())
    if valid_count < 2:
        return astar_path.new_tensor(0.0, dtype=torch.float32)

    p0 = astar_path[0].float()
    p1 = astar_path[1].float()
    delta = p1 - p0
    if torch.norm(delta) <= 0.0:
        return astar_path.new_tensor(0.0, dtype=torch.float32)
    return torch.atan2(delta[1], delta[0])


def commands_to_training_target(commands: torch.Tensor, *, velocity_scale: float) -> torch.Tensor:
    target = commands.new_zeros((commands.shape[0], 4), dtype=torch.float32)
    target[:, 0:2] = commands[:, 0:2] / float(velocity_scale)
    target[:, 2] = torch.sin(commands[:, 2])
    target[:, 3] = torch.cos(commands[:, 2])
    return target


def training_target_to_commands(target: torch.Tensor, *, velocity_scale: float) -> torch.Tensor:
    commands = target.new_zeros((*target.shape[:-1], 3))
    commands[..., 0:2] = target[..., 0:2] * float(velocity_scale)
    sin_heading = target[..., 2]
    cos_heading = target[..., 3]
    norm = torch.clamp(torch.sqrt(sin_heading.square() + cos_heading.square()), min=1e-6)
    commands[..., 2] = torch.atan2(sin_heading / norm, cos_heading / norm)
    return commands


def integrate_velocity_commands_xy(start_xy: torch.Tensor, commands: torch.Tensor, *, dt: float) -> torch.Tensor:
    pos = start_xy.float().clone()
    points = [pos.clone()]
    for command in commands.float():
        vx = command[0]
        vy = command[1]
        heading = command[2]
        if not bool(torch.isfinite(heading).item()):
            heading = heading.new_tensor(0.0)
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        delta = torch.stack([
            cos_h * vx - sin_h * vy,
            sin_h * vx + cos_h * vy,
        ]) * float(dt)
        pos = pos + delta
        points.append(pos.clone())
    return torch.stack(points)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=pred.dtype).unsqueeze(-1)
    denom = torch.clamp(valid.sum() * pred.shape[-1], min=1.0)
    return ((pred - target).square() * valid).sum() / denom


def masked_velocity_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.to(dtype=pred.dtype).unsqueeze(-1)
    denom = torch.clamp(valid.sum() * 2.0, min=1.0)
    return ((pred[..., :2] - target[..., :2]).square() * valid).sum() / denom


def masked_heading_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_angle = torch.atan2(pred[..., 2], pred[..., 3])
    target_angle = torch.atan2(target[..., 2], target[..., 3])
    error = torch.remainder(pred_angle - target_angle + math.pi, 2.0 * math.pi) - math.pi
    valid = mask.to(dtype=pred.dtype)
    return (torch.abs(error) * valid).sum() / torch.clamp(valid.sum(), min=1.0)


def _crop_xy_tensor(tensor: torch.Tensor, start_grid: torch.Tensor, *, crop_size: int, pad_value: float) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError(f"Expected tensor with at least 2 dims, got {tuple(tensor.shape)}")

    half = crop_size // 2
    x = int(start_grid[0].item())
    y = int(start_grid[1].item())
    x0 = x - half
    y0 = y - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    pad_left = max(0, -y0)
    pad_right = max(0, y1 - tensor.shape[1])
    pad_top = max(0, -x0)
    pad_bottom = max(0, x1 - tensor.shape[0])

    x0_clamped = max(0, x0)
    x1_clamped = min(tensor.shape[0], x1)
    y0_clamped = max(0, y0)
    y1_clamped = min(tensor.shape[1], y1)
    cropped = tensor[x0_clamped:x1_clamped, y0_clamped:y1_clamped]

    if pad_left or pad_right or pad_top or pad_bottom:
        permute_order = (tuple(range(2, cropped.dim())) + (0, 1))
        inverse_order = tuple(range(cropped.dim() - 2, cropped.dim())) + tuple(range(cropped.dim() - 2))
        channels_first = cropped.permute(permute_order).unsqueeze(0)
        padded = F.pad(
            channels_first,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=float(pad_value),
        ).squeeze(0)
        cropped = padded.permute(inverse_order)

    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        raise RuntimeError(f"Crop produced shape {tuple(cropped.shape)} for crop_size={crop_size}")
    return cropped
