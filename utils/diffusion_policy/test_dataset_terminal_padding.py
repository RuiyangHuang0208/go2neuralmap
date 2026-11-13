from __future__ import annotations

import json
from typing import Optional

import torch

from utils.diffusion_policy.dataset import (
    AugmentedNeuralMapDataset,
    AugmentedNeuralMapDatasetConfig,
    TerminalCommandPaddingConfig,
    training_target_to_commands,
)


def test_terminal_padding_fills_invalid_tail_after_reaching_goal(tmp_path):
    _write_dataset(tmp_path, commands=torch.tensor([[[1.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))

    dataset = AugmentedNeuralMapDataset(
        AugmentedNeuralMapDatasetConfig(
            dataset_dir=str(tmp_path),
            crop_size=2,
            terminal_padding=TerminalCommandPaddingConfig(
                enabled=True,
                near_threshold=1.0,
                goal_tolerance=0.35,
                only_if_reached_goal=True,
            ),
        )
    )

    sample = dataset[0]
    commands = training_target_to_commands(sample["target"], velocity_scale=dataset.cfg.velocity_scale)

    assert sample["target_mask"].tolist() == [True, True, True]
    assert torch.allclose(commands[:, 0:2], torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    assert torch.allclose(commands[:, 2], torch.full((3,), 0.5))


def test_terminal_padding_does_not_fill_when_goal_not_reached(tmp_path):
    _write_dataset(tmp_path, goal_xy=torch.tensor([[0.9, 0.0]]), commands=torch.tensor([[[0.1, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]))

    dataset = AugmentedNeuralMapDataset(
        AugmentedNeuralMapDatasetConfig(
            dataset_dir=str(tmp_path),
            crop_size=2,
            terminal_padding=TerminalCommandPaddingConfig(
                enabled=True,
                near_threshold=1.0,
                goal_tolerance=0.35,
                only_if_reached_goal=True,
            ),
        )
    )

    sample = dataset[0]

    assert sample["target_mask"].tolist() == [True, False, False]


def _write_dataset(
    tmp_path,
    *,
    goal_xy: Optional[torch.Tensor] = None,
    commands: torch.Tensor,
) -> None:
    metadata = {
        "schema_version": "cropped_v1",
        "latent_dim": 1,
        "collector": {
            "velocity_command_dt": 0.5,
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    shard = {
        "map_crop": torch.zeros(1, 4, 2, 2),
        "uncertainty_crop": torch.zeros(1, 4, 2, 2),
        "current_latent": torch.zeros(1, 4),
        "start_grid": torch.zeros(1, 2, dtype=torch.long),
        "start_xy": torch.zeros(1, 2),
        "goal_xy": torch.tensor([[0.5, 0.0]]) if goal_xy is None else goal_xy,
        "initial_heading": torch.zeros(1),
        "velocity_commands": commands,
        "velocity_command_mask": torch.tensor([[True, False, False]]),
    }
    torch.save(shard, tmp_path / "shard_000000.pt")
