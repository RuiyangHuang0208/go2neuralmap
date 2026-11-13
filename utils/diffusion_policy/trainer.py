from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler, Subset, random_split

from .dataset import AugmentedNeuralMapDataset, AugmentedNeuralMapDatasetConfig
from .registry import build_algorithm


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-6


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "go2_neural_map"
    entity: Optional[str] = None
    name: Optional[str] = None
    mode: Optional[str] = None


@dataclass
class ResumeConfig:
    enabled: bool = False
    checkpoint_path: Optional[str] = None
    strict: bool = True
    load_optimizer: bool = True


@dataclass
class TrainingConfig:
    seed: int = 0
    device: str = "cuda"
    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 0
    train_split: float = 0.9
    log_interval: int = 20
    eval_interval: int = 1
    checkpoint_interval: int = 1
    output_dir: str = "logs/diffusion_policy"
    dataset: AugmentedNeuralMapDatasetConfig = field(default_factory=AugmentedNeuralMapDatasetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)


def train_from_config(cfg_omega) -> None:
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(TrainingConfig), cfg_omega.training))
    if str(cfg_omega.algorithm.name) == "guided_joint_ddpm":
        cfg.dataset.require_cost_map = True
        OmegaConf.update(cfg_omega, "training.dataset.require_cost_map", True, merge=True)
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg_omega, output_dir / "config.yaml")

    dataset = AugmentedNeuralMapDataset(cfg.dataset)
    train_set, val_set = split_dataset(dataset, float(cfg.train_split), int(cfg.seed))
    logger.info("Start creating train loader")
    train_loader = build_dataloader(
        train_set,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        seed=int(cfg.seed),
        shuffle=True,
    )
    logger.info("Finished creating train loader")
    logger.info("Start creating val loader")
    val_loader = build_dataloader(
        val_set,
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.num_workers),
        pin_memory=device.type == "cuda",
        seed=int(cfg.seed),
        shuffle=False,
    )
    logger.info("Finished creating val loader")

    algorithm = build_algorithm(cfg_omega.algorithm).to(device)
    optimizer = torch.optim.AdamW(
        algorithm.model.parameters(),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    start_epoch, global_step = maybe_resume_training(cfg, algorithm, optimizer, device)
    wandb_run = maybe_init_wandb(cfg, cfg_omega)

    logger.info("Training on {} with {} train and {} val samples", device, len(train_set), len(val_set))
    for epoch in range(start_epoch, int(cfg.epochs)):
        train_metrics = MetricAverager()
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            metrics = algorithm.train_step(batch, optimizer, global_step=global_step)
            train_metrics.update(metrics)
            if batch_idx % int(cfg.log_interval) == 0:
                logger.info("epoch={} batch={} train={}", epoch, batch_idx, compact_metrics(metrics))
            if wandb_run is not None:
                wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=global_step)
            global_step += 1

        logger.info("epoch={} train_avg={}", epoch, compact_metrics(train_metrics.compute()))

        if (epoch + 1) % int(cfg.eval_interval) == 0:
            val_metrics = evaluate(algorithm, val_loader, device)
            logger.info("epoch={} val={}", epoch, compact_metrics(val_metrics))
            if wandb_run is not None:
                wandb_run.log({f"val/{key}": value for key, value in val_metrics.items()}, step=global_step)

        if (epoch + 1) % int(cfg.checkpoint_interval) == 0:
            save_checkpoint(output_dir, epoch, global_step, algorithm, optimizer)

    save_checkpoint(output_dir, int(cfg.epochs) - 1, global_step, algorithm, optimizer, filename="latest.pt")
    if wandb_run is not None:
        wandb_run.finish()


@torch.no_grad()
def evaluate(algorithm, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    metrics = MetricAverager()
    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        metrics.update(algorithm.eval_step(batch))
    return metrics.compute()


def split_dataset(dataset: AugmentedNeuralMapDataset, train_split: float, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be in (0, 1)")
    train_len = int(len(dataset) * train_split)
    val_len = len(dataset) - train_len
    if train_len <= 0 or val_len <= 0:
        raise ValueError(f"Dataset too small for train_split={train_split}: len={len(dataset)}")
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_len, val_len], generator=generator)


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    if isinstance(dataset, Subset) and isinstance(dataset.dataset, AugmentedNeuralMapDataset):
        return DataLoader(
            dataset,
            batch_sampler=ShardBatchSampler(dataset, batch_size=batch_size, seed=seed, shuffle=shuffle),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class ShardBatchSampler(Sampler[List[int]]):
    """Batch Subset positions by shard to avoid repeatedly loading large shards."""

    def __init__(self, subset: Subset, *, batch_size: int, seed: int, shuffle: bool):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(subset.dataset, AugmentedNeuralMapDataset):
            raise TypeError("ShardBatchSampler requires a Subset of AugmentedNeuralMapDataset")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.groups: List[List[int]] = []

        positions_by_shard: Dict[int, List[int]] = {}
        for subset_pos, dataset_idx in enumerate(subset.indices):
            shard_idx, _ = subset.dataset.index[int(dataset_idx)]
            positions_by_shard.setdefault(int(shard_idx), []).append(int(subset_pos))
        for shard_idx in sorted(positions_by_shard):
            self.groups.append(positions_by_shard[shard_idx])

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1

        group_order = list(range(len(self.groups)))
        if self.shuffle and len(group_order) > 1:
            group_order = torch.randperm(len(group_order), generator=generator).tolist()

        for group_idx in group_order:
            group = self.groups[int(group_idx)]
            if self.shuffle and len(group) > 1:
                order = torch.randperm(len(group), generator=generator).tolist()
                group = [group[int(idx)] for idx in order]

            for start in range(0, len(group), self.batch_size):
                yield group[start:start + self.batch_size]

    def __len__(self) -> int:
        return sum((len(group) + self.batch_size - 1) // self.batch_size for group in self.groups)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def maybe_resume_training(
    cfg: TrainingConfig,
    algorithm,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, int]:
    if not cfg.resume.enabled:
        return 0, 0
    if cfg.resume.checkpoint_path is None:
        raise ValueError("training.resume.checkpoint_path must be set when resume.enabled=True")

    checkpoint_path = Path(cfg.resume.checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "algorithm" not in checkpoint:
        raise KeyError(f"{checkpoint_path} does not contain an algorithm state")
    algorithm.load_state_dict(checkpoint["algorithm"], strict=bool(cfg.resume.strict))

    if cfg.resume.load_optimizer:
        if "optimizer" not in checkpoint:
            raise KeyError(f"{checkpoint_path} does not contain an optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    logger.info(
        "Resumed training from {} at start_epoch={} global_step={} load_optimizer={}",
        checkpoint_path,
        start_epoch,
        global_step,
        bool(cfg.resume.load_optimizer),
    )
    return start_epoch, global_step


def maybe_init_wandb(cfg: TrainingConfig, cfg_omega):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb.enabled=True but wandb is not installed; continuing without wandb")
        return None

    from datetime import datetime
    run_name = cfg.wandb.name if \
        cfg.wandb.name is not None \
        else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    kwargs = {
        "project": cfg.wandb.project,
        "name": run_name,
        "config": OmegaConf.to_container(cfg_omega, resolve=True),
    }
    if cfg.wandb.entity is not None:
        kwargs["entity"] = cfg.wandb.entity
    if cfg.wandb.mode is not None:
        kwargs["mode"] = cfg.wandb.mode
    return wandb.init(**kwargs)


def save_checkpoint(output_dir: Path, epoch: int, global_step: int, algorithm, optimizer, filename: Optional[str] = None) -> None:
    if filename is None:
        filename = f"checkpoint_epoch_{epoch:04d}.pt"
    path = output_dir / filename
    torch.save(
        {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "algorithm": algorithm.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    logger.info("Saved checkpoint {}", path)


class MetricAverager:
    def __init__(self):
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def compute(self) -> Dict[str, float]:
        return {key: self.totals[key] / max(self.counts[key], 1) for key in self.totals}


def compact_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.5f}" for key, value in sorted(metrics.items()))
