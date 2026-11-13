import pytest
import torch

from utils.cost_map import (
    DirectionalEMATransitionCostMap,
    DirectionalEMATransitionCostMapCfg,
)


def make_cfg(**kwargs) -> DirectionalEMATransitionCostMapCfg:
    base_kwargs = {
        "x_range": (0.0, 3.0),
        "y_range": (0.0, 3.0),
        "device": "cpu",
        "dtype": "torch.float32",
        "least_count": 1.0,
        "init_value": 10.0,
        "alpha": 0.5,
    }
    base_kwargs.update(kwargs)
    return DirectionalEMATransitionCostMapCfg(**base_kwargs)


def test_directional_cost_map_updates_pos_x_edge_with_ema():
    cost_map = DirectionalEMATransitionCostMap(num_envs=1, cfg=make_cfg())

    cost_map.update(torch.tensor([[0.2, 0.2]]), torch.tensor([6.0]))
    moved_mask = cost_map.update(torch.tensor([[1.2, 0.2]]), torch.tensor([2.0]))

    assert torch.equal(moved_mask, torch.tensor([True]))
    assert cost_map.maps[0, 0, 0, 0].item() == pytest.approx(8.0)
    assert cost_map.get_traversal_cost((0, 0), (1, 0), env_id=0) == pytest.approx(8.0)


def test_directional_cost_map_updates_each_cardinal_direction():
    cost_map = DirectionalEMATransitionCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=0.0))

    transitions = [
        ((1.2, 1.2), 1.0),
        ((2.2, 1.2), 2.0),
        ((1.2, 1.2), 3.0),
        ((1.2, 2.2), 4.0),
        ((1.2, 1.2), 5.0),
    ]
    for pos, cost in transitions:
        cost_map.update(torch.tensor([pos]), torch.tensor([cost]))

    assert cost_map.maps[0, 1, 1, 0].item() == pytest.approx(1.0)
    assert cost_map.maps[0, 2, 1, 1].item() == pytest.approx(2.0)
    assert cost_map.maps[0, 1, 1, 2].item() == pytest.approx(3.0)
    assert cost_map.maps[0, 1, 2, 3].item() == pytest.approx(4.0)


def test_directional_cost_map_skips_diagonal_transition_updates():
    cost_map = DirectionalEMATransitionCostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=9.0))

    cost_map.update(torch.tensor([[0.2, 0.2]]), torch.tensor([5.0]))
    moved_mask = cost_map.update(torch.tensor([[1.2, 1.2]]), torch.tensor([7.0]))

    assert torch.equal(moved_mask, torch.tensor([False]))
    assert torch.all(cost_map.maps[0, 0, 0] == 9.0)
