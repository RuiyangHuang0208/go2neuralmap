from __future__ import annotations

import json
import math

import pytest
import torch

from utils.diffusion_policy.dataset import AugmentedNeuralMapDataset, AugmentedNeuralMapDatasetConfig


def test_dataset_can_use_state_global_heading_for_current_heading(tmp_path):
    _write_dataset(tmp_path, include_state_global_heading=True)

    dataset = AugmentedNeuralMapDataset(
        AugmentedNeuralMapDatasetConfig(
            dataset_dir=str(tmp_path),
            crop_size=2,
            current_heading_source="state_global_heading",
        )
    )

    sample = dataset[0]

    assert torch.allclose(sample["current_heading"], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(sample["initial_heading"], torch.tensor(0.0))


def test_dataset_requires_state_global_heading_when_configured(tmp_path):
    _write_dataset(tmp_path, include_state_global_heading=False)

    with pytest.raises(ValueError, match="state_global_heading"):
        AugmentedNeuralMapDataset(
            AugmentedNeuralMapDatasetConfig(
                dataset_dir=str(tmp_path),
                crop_size=2,
                current_heading_source="state_global_heading",
            )
        )


def _write_dataset(tmp_path, *, include_state_global_heading: bool) -> None:
    metadata = {
        "schema_version": "cropped_v1",
        "latent_dim": 1,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    shard = {
        "map_crop": torch.zeros(1, 4, 2, 2),
        "uncertainty_crop": torch.zeros(1, 4, 2, 2),
        "current_latent": torch.zeros(1, 4),
        "start_grid": torch.zeros(1, 2, dtype=torch.long),
        "start_xy": torch.zeros(1, 2),
        "goal_xy": torch.ones(1, 2),
        "initial_heading": torch.zeros(1),
        "velocity_commands": torch.zeros(1, 3, 3),
        "velocity_command_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    if include_state_global_heading:
        shard["state_global_heading"] = torch.full((1,), math.pi / 2.0)
    torch.save(shard, tmp_path / "shard_000000.pt")
