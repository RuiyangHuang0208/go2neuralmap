from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from utils.diffusion_policy.dataset import (
    AugmentedNeuralMapDataset,
    AugmentedNeuralMapDatasetConfig,
    masked_heading_error,
    masked_velocity_mse,
    training_target_to_commands,
)
from utils.diffusion_policy.registry import build_algorithm


DEFAULT_DATASET_DIR = (
    "logs/2026-05-18_d00496e/demo_plane_real_path_dataset_5env_1_14-51-20/dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one sample exactly as AugmentedNeuralMapDataset feeds it to diffusion training."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--run-dir", default=None, help="Optional diffusion run dir containing config.yaml and latest.pt.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to RUN_DIR/latest.pt.")
    parser.add_argument("--index", type=int, default=-1, help="Dataset item index. Negative indices are supported.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of diffusion samples to draw when --run-dir is set.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--current-heading-source", default=None, choices=["initial_heading", "state_global_heading"])
    parser.add_argument("--output", default=None, help="Optional image output path. If omitted, show an interactive window.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = resolve_device(args.device)

    run_cfg = load_run_config(args.run_dir)
    dataset_cfg = build_dataset_config(args, run_cfg)
    dataset = AugmentedNeuralMapDataset(dataset_cfg)
    sample = dataset[int(args.index)]
    raw_sample, shard_idx, local_idx = load_raw_sample(dataset, int(args.index))

    batch = tensor_batch(sample, device)
    label_commands = training_target_to_commands(
        batch["target"],
        velocity_scale=float(dataset_cfg.velocity_scale),
    )[0].detach().cpu()
    target_mask = batch["target_mask"][0].detach().cpu().bool()

    predicted_commands = None
    metric_text = []
    if run_cfg is not None:
        algorithm = load_algorithm(run_cfg, args, device)
        predictions = []
        pred_targets = []
        for _ in range(max(1, int(args.num_samples))):
            target = sample_target(algorithm, batch)
            pred_targets.append(target.detach().cpu())
            commands = training_target_to_commands(
                target,
                velocity_scale=float(dataset_cfg.velocity_scale),
            )[0]
            predictions.append(commands.detach().cpu())
        predicted_commands = torch.stack(predictions)
        pred_targets_t = torch.cat(pred_targets, dim=0)
        target_t = batch["target"].detach().cpu().expand_as(pred_targets_t)
        mask_t = batch["target_mask"].detach().cpu().expand(pred_targets_t.shape[0], -1)
        metric_text.extend([
            f"sample_velocity_mse={masked_velocity_mse(pred_targets_t, target_t, mask_t).item():.6f}",
            f"sample_heading_error={masked_heading_error(pred_targets_t, target_t, mask_t).item():.6f} rad",
        ])

    fig = plot_sample(
        raw_sample=raw_sample,
        processed_sample=sample,
        label_commands=label_commands,
        target_mask=target_mask,
        predicted_commands=predicted_commands,
        dataset_cfg=dataset_cfg,
        title=f"dataset index={args.index} shard={shard_idx} local={local_idx}",
        metric_text=metric_text,
    )

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


def resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_run_config(run_dir: Optional[str]):
    if run_dir is None:
        return None
    config_path = Path(run_dir).expanduser() / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    return OmegaConf.load(config_path)


def build_dataset_config(args: argparse.Namespace, run_cfg) -> AugmentedNeuralMapDatasetConfig:
    if run_cfg is not None:
        cfg = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(AugmentedNeuralMapDatasetConfig),
                run_cfg.training.dataset,
            )
        )
    else:
        cfg = AugmentedNeuralMapDatasetConfig()

    cfg.dataset_dir = str(Path(args.dataset_dir).expanduser())
    if args.current_heading_source is not None:
        cfg.current_heading_source = str(args.current_heading_source)
    return cfg


def load_algorithm(run_cfg, args: argparse.Namespace, device: torch.device):
    algorithm = build_algorithm(run_cfg.algorithm).to(device)
    checkpoint_path = Path(args.checkpoint).expanduser() if args.checkpoint else Path(args.run_dir).expanduser() / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algorithm_state = checkpoint.get("algorithm", checkpoint)
    algorithm.load_state_dict(algorithm_state, strict=True)
    set_eval(algorithm)
    return algorithm


def set_eval(algorithm) -> None:
    if hasattr(algorithm, "model"):
        algorithm.model.eval()
    if hasattr(algorithm, "target_model"):
        algorithm.target_model.eval()


def sample_target(algorithm, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    horizon = int(batch["target"].shape[1])
    if hasattr(algorithm, "infer"):
        return algorithm.infer(batch, horizon)
    if hasattr(algorithm, "sample"):
        return algorithm.sample(batch, horizon)
    return algorithm.model(batch)


def tensor_batch(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }


def load_raw_sample(dataset: AugmentedNeuralMapDataset, index: int) -> Tuple[Dict[str, torch.Tensor], int, int]:
    shard_idx, local_idx = dataset.index[int(index)]
    shard = dataset._load_shard(shard_idx)
    raw_sample = {
        key: value[int(local_idx)].detach().cpu()
        for key, value in shard.items()
        if torch.is_tensor(value) and int(value.shape[0]) > int(local_idx)
    }
    return raw_sample, int(shard_idx), int(local_idx)


def plot_sample(
    *,
    raw_sample: Dict[str, torch.Tensor],
    processed_sample: Dict[str, torch.Tensor],
    label_commands: torch.Tensor,
    target_mask: torch.Tensor,
    predicted_commands: Optional[torch.Tensor],
    dataset_cfg: AugmentedNeuralMapDatasetConfig,
    title: str,
    metric_text: list[str],
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title)

    valid_label = label_commands[target_mask]
    start_grid = raw_sample.get("start_grid", torch.tensor([0, 0])).float()
    goal_grid = raw_sample.get("goal_grid", None)
    astar_path = raw_sample.get("astar_path", None)
    astar_mask = raw_sample.get("astar_path_mask", None)
    cost_crop = raw_sample.get("cost_map_crop", None)

    label_crop_path = integrate_commands_in_cells(
        valid_label,
        start_cell=torch.tensor([float(dataset_cfg.crop_size // 2), float(dataset_cfg.crop_size // 2)]),
        least_count=float(metadata_value(raw_sample, "least_count", 0.25)),
        dt=float(metadata_value(raw_sample, "velocity_command_dt", 0.5)),
    )
    label_global_path = label_crop_path - float(dataset_cfg.crop_size // 2) + start_grid.view(1, 2)

    ax = axes[0]
    if cost_crop is not None:
        cost_image = cost_crop.float()
        if cost_image.dim() == 3:
            cost_image = cost_image.mean(dim=0)
        im = ax.imshow(cost_image, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(label_crop_path[:, 1], label_crop_path[:, 0], color="tab:blue", linewidth=2.5, label="label")
    if predicted_commands is not None:
        for commands in predicted_commands:
            pred_path = integrate_commands_in_cells(
                commands,
                start_cell=torch.tensor([float(dataset_cfg.crop_size // 2), float(dataset_cfg.crop_size // 2)]),
                least_count=float(metadata_value(raw_sample, "least_count", 0.25)),
                dt=float(metadata_value(raw_sample, "velocity_command_dt", 0.5)),
            )
            ax.plot(pred_path[:, 1], pred_path[:, 0], color="tab:red", alpha=0.25)
    center = float(dataset_cfg.crop_size // 2)
    ax.scatter([center], [center], color="black", s=40, zorder=3, label="start")
    ax.set_title("local crop path")
    ax.set_aspect("equal")
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(label_global_path[:, 1], label_global_path[:, 0], color="tab:blue", linewidth=2.5, label="label")
    if predicted_commands is not None:
        for commands in predicted_commands:
            pred_crop_path = integrate_commands_in_cells(
                commands,
                start_cell=torch.tensor([center, center]),
                least_count=float(metadata_value(raw_sample, "least_count", 0.25)),
                dt=float(metadata_value(raw_sample, "velocity_command_dt", 0.5)),
            )
            pred_global_path = pred_crop_path - center + start_grid.view(1, 2)
            ax.plot(pred_global_path[:, 1], pred_global_path[:, 0], color="tab:red", alpha=0.25)
    if astar_path is not None and astar_mask is not None:
        valid_astar = astar_path[astar_mask.bool()]
        if valid_astar.numel() > 0:
            ax.plot(valid_astar[:, 1], valid_astar[:, 0], color="black", linestyle="--", linewidth=1.2, label="astar")
    ax.scatter(start_grid[1], start_grid[0], color="black", s=40, zorder=3, label="start")
    if goal_grid is not None:
        ax.scatter(goal_grid[1], goal_grid[0], color="lime", edgecolors="black", s=60, zorder=3, label="goal")
    ax.invert_yaxis()
    ax.set_title("global grid path")
    ax.set_aspect("equal")
    ax.legend(loc="best")

    ax = axes[2]
    steps = torch.arange(label_commands.shape[0])
    ax.plot(steps, label_commands[:, 0], label="label vx", color="tab:blue")
    ax.plot(steps, label_commands[:, 1], label="label vy", color="tab:cyan")
    ax.plot(steps, label_commands[:, 2], label="label heading", color="tab:green")
    if predicted_commands is not None:
        mean_pred = predicted_commands.mean(dim=0)
        ax.plot(steps, mean_pred[:, 0], "--", label="pred mean vx", color="tab:red")
        ax.plot(steps, mean_pred[:, 1], "--", label="pred mean vy", color="tab:orange")
        ax.plot(steps, mean_pred[:, 2], "--", label="pred mean heading", color="tab:purple")
    ax.set_title("command sequence")
    ax.set_xlabel("horizon step")
    ax.legend(loc="best", fontsize=8)

    details = [
        f"current_heading=[sin,cos]={processed_sample['current_heading'].tolist()}",
        f"initial_heading={float(processed_sample['initial_heading'].item()):.4f}",
        f"goal_delta={processed_sample['goal_delta'].tolist()}",
        *metric_text,
    ]
    fig.text(0.01, 0.01, "\n".join(details), fontsize=9, family="monospace")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def integrate_commands_in_cells(
    commands: torch.Tensor,
    *,
    start_cell: torch.Tensor,
    least_count: float,
    dt: float,
) -> torch.Tensor:
    pos = start_cell.float().clone()
    points = [pos.clone()]
    for command in commands:
        vx = float(command[0].item())
        vy = float(command[1].item())
        heading = float(command[2].item())
        if not math.isfinite(heading):
            heading = 0.0
        dx_m = (math.cos(heading) * vx - math.sin(heading) * vy) * float(dt)
        dy_m = (math.sin(heading) * vx + math.cos(heading) * vy) * float(dt)
        pos = pos + torch.tensor([dx_m / least_count, dy_m / least_count])
        points.append(pos.clone())
    return torch.stack(points)


def metadata_value(raw_sample: Dict[str, torch.Tensor], key: str, default: float) -> float:
    value = raw_sample.get(key, None)
    if value is None:
        return float(default)
    return float(value.item())


if __name__ == "__main__":
    main()
