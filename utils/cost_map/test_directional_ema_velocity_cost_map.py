import pytest
import torch

from utils.cost_map import (
    DirectionalEMAVelocityCostMap,
    DirectionalEMAVelocityCostMapCfg,
)


def make_cfg(**kwargs) -> DirectionalEMAVelocityCostMapCfg:
    base_kwargs = {
        "x_range": (0.0, 3.0),
        "y_range": (0.0, 3.0),
        "device": "cpu",
        "dtype": "torch.float32",
        "least_count": 1.0,
        "init_value": 10.0,
        "alpha": 0.5,
        "stuck_threshold_steps": 2,
        "min_velocity_norm": 1e-3,
        "min_command_norm": 1e-3,
    }
    base_kwargs.update(kwargs)
    return DirectionalEMAVelocityCostMapCfg(**base_kwargs)


def test_velocity_cost_map_updates_both_axes_from_root_velocity():
    cost_map = DirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    valid_mask = cost_map.update(
        current_pos=torch.tensor([[1.2, 1.2]]),
        new_values=torch.tensor([6.0]),
        root_velocity=torch.tensor([[0.5, 0.1]]),
        command=torch.tensor([[0.0, 1.0]]),
    )

    assert torch.equal(valid_mask, torch.tensor([True]))
    assert cost_map.maps[0, 1, 1, 0].item() == pytest.approx(5.0)
    assert cost_map.maps[0, 1, 1, 2].item() == pytest.approx(1.0)


def test_velocity_cost_map_splits_update_evenly_near_diagonal():
    cost_map = DirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    valid_mask = cost_map.update(
        current_pos=torch.tensor([[1.2, 1.2]]),
        new_values=torch.tensor([8.0]),
        root_velocity=torch.tensor([[1.0, 1.0]]),
        command=torch.tensor([[0.0, 0.0]]),
    )

    assert torch.equal(valid_mask, torch.tensor([True]))
    assert cost_map.maps[0, 1, 1, 0].item() == pytest.approx(4.0)
    assert cost_map.maps[0, 1, 1, 2].item() == pytest.approx(4.0)


def test_velocity_cost_map_uses_command_heading_after_stuck_threshold():
    cost_map = DirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0, stuck_threshold_steps=1))
    current_pos = torch.tensor([[1.2, 1.2]])

    cost_map.update(
        current_pos=current_pos,
        new_values=torch.tensor([3.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, -1.0]]),
    )
    cost_map.update(
        current_pos=current_pos,
        new_values=torch.tensor([4.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, -1.0]]),
    )
    cost_map.update(
        current_pos=current_pos,
        new_values=torch.tensor([7.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, -1.0]]),
    )

    assert cost_map.maps[0, 1, 1, 0].item() == pytest.approx(4.0)
    assert cost_map.maps[0, 1, 1, 3].item() == pytest.approx(7.0)


def test_velocity_cost_map_falls_back_to_command_when_velocity_is_too_small():
    cost_map = DirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    valid_mask = cost_map.update(
        current_pos=torch.tensor([[0.2, 0.2]]),
        new_values=torch.tensor([5.0]),
        root_velocity=torch.tensor([[0.0, 0.0]]),
        command=torch.tensor([[-0.2, 0.0]]),
    )

    assert torch.equal(valid_mask, torch.tensor([True]))
    assert cost_map.maps[0, 0, 0, 1].item() == pytest.approx(5.0)


def test_velocity_cost_map_skips_update_when_no_valid_heading_exists():
    cost_map = DirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=9.0))

    valid_mask = cost_map.update(
        current_pos=torch.tensor([[0.2, 0.2]]),
        new_values=torch.tensor([5.0]),
        root_velocity=torch.tensor([[0.0, 0.0]]),
        command=torch.tensor([[0.0, 0.0]]),
    )

    assert torch.equal(valid_mask, torch.tensor([False]))
    assert torch.all(cost_map.maps[0, 0, 0] == 9.0)


def test_velocity_cost_map_reset_can_preserve_map_values_while_clearing_running_state():
    cost_map = DirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=10.0))
    current_pos = torch.tensor([[1.2, 1.2]])

    cost_map.update(
        current_pos=current_pos,
        new_values=torch.tensor([6.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, 0.0]]),
    )
    cost_map.update(
        current_pos=current_pos,
        new_values=torch.tensor([8.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, 0.0]]),
    )

    cost_map.reset(torch.tensor([1]), reset_values=False)

    assert cost_map.maps[0, 1, 1, 0].item() == pytest.approx(8.0)
    assert not bool(cost_map._has_last_indices[0].item())
    assert cost_map._steps_in_current_cell[0].item() == 0
