#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.diffusion_policy.dataset import AugmentedNeuralMapDataset
from utils.diffusion_policy.registry import build_algorithm
from utils.diffusion_policy.trainer import TrainingConfig, evaluate, split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion policy checkpoint.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training output directory containing config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to <run_dir>/latest.pt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config path. Defaults to <run_dir>/config.yaml.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "train", "all"),
        default="val",
        help="Dataset split to evaluate. Defaults to the validation split used during training.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader num_workers.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda or cpu.")
    return parser.parse_args()


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("Requested {} but CUDA is unavailable; using CPU", requested)
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_algorithm(cfg_omega, checkpoint_path: Path, device: torch.device):
    algorithm = build_algorithm(cfg_omega.algorithm).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    algorithm_state = checkpoint.get("algorithm", checkpoint)
    if "model" not in algorithm_state:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain algorithm.model state")
    algorithm.model.load_state_dict(algorithm_state["model"])
    logger.info(
        "Loaded checkpoint {} epoch={} global_step={}",
        checkpoint_path,
        checkpoint.get("epoch", "unknown"),
        checkpoint.get("global_step", "unknown"),
    )
    return algorithm


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    config_path = (args.config or run_dir / "config.yaml").expanduser()
    checkpoint_path = (args.checkpoint or run_dir / "latest.pt").expanduser()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg_omega = OmegaConf.load(config_path)
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(TrainingConfig), cfg_omega.training))
    device = resolve_device(args.device or str(cfg.device))

    dataset = AugmentedNeuralMapDataset(cfg.dataset)
    if args.split == "all":
        eval_set = dataset
    else:
        train_set, val_set = split_dataset(dataset, float(cfg.train_split), int(cfg.seed))
        eval_set = train_set if args.split == "train" else val_set

    batch_size = int(args.batch_size or cfg.batch_size)
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.num_workers)
    loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    algorithm = load_algorithm(cfg_omega, checkpoint_path, device)
    logger.info(
        "Evaluating split={} samples={} batches={} batch_size={} device={}",
        args.split,
        len(eval_set),
        len(loader),
        batch_size,
        device,
    )
    metrics = evaluate(algorithm, loader, device)
    for key, value in sorted(metrics.items()):
        logger.info("{}={:.6f}", key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
