import math

import pytest
import torch

from utils.cost_map import (
    CollisionDirectionalEMAVelocityCostMap,
    CollisionDirectionalEMAVelocityCostMapCfg,
)


def make_cfg(**kwargs) -> CollisionDirectionalEMAVelocityCostMapCfg:
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
        "impassable_cost": 999.0,
    }
    base_kwargs.update(kwargs)
    return CollisionDirectionalEMAVelocityCostMapCfg(**base_kwargs)


def yaw_to_quat_xyzw(yaw: float) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]], dtype=torch.float32)


def test_front_collision_blocks_forward_world_x_edge():
    cost_map = CollisionDirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    valid_mask = cost_map.update(
        current_pos=torch.tensor([[1.2, 1.2]]),
        new_values=torch.tensor([5.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, 0.0]]),
        directional_collision=torch.tensor([[True, False, False, False]]),
        base_quat=yaw_to_quat_xyzw(0.0),
    )

    assert torch.equal(valid_mask, torch.tensor([True]))
    assert cost_map.maps[0, 1, 1, cost_map.POS_X].item() == pytest.approx(999.0)


def test_front_collision_blocks_forward_world_y_edge_when_facing_left():
    cost_map = CollisionDirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    cost_map.update(
        current_pos=torch.tensor([[1.2, 1.2]]),
        new_values=torch.tensor([5.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, 0.0]]),
        directional_collision=torch.tensor([[True, False, False, False]]),
        base_quat=yaw_to_quat_xyzw(math.pi / 2.0),
    )

    assert cost_map.maps[0, 1, 1, cost_map.POS_Y].item() == pytest.approx(999.0)


def test_left_collision_blocks_positive_y_edge_when_facing_world_x():
    cost_map = CollisionDirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    cost_map.update(
        current_pos=torch.tensor([[1.2, 1.2]]),
        new_values=torch.tensor([5.0]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
        command=torch.tensor([[0.0, 0.0]]),
        directional_collision=torch.tensor([[False, True, False, False]]),
        base_quat=yaw_to_quat_xyzw(0.0),
    )

    assert cost_map.maps[0, 1, 1, cost_map.POS_Y].item() == pytest.approx(999.0)


def test_collision_blocking_overrides_normal_ema_update():
    cost_map = CollisionDirectionalEMAVelocityCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    cost_map.update(
        current_pos=torch.tensor([[1.2, 1.2]]),
        new_values=torch.tensor([6.0]),
        root_velocity=torch.tensor([[0.8, 0.2]]),
        command=torch.tensor([[0.0, 0.0]]),
        directional_collision=torch.tensor([[True, False, False, False]]),
        base_quat=yaw_to_quat_xyzw(0.0),
    )

    assert cost_map.maps[0, 1, 1, cost_map.POS_X].item() == pytest.approx(999.0)
        # The secondary Y edge still follows the standard directional EMA update.
    assert cost_map.maps[0, 1, 1, cost_map.POS_Y].item() == pytest.approx(1.2)
