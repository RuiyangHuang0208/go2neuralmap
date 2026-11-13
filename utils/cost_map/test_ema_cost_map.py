import pytest
import torch

from utils.cost_map import EMACostMap, EMACostMapCfg


def make_cfg(**kwargs) -> EMACostMapCfg:
    base_kwargs = {
        "x_range": (0.0, 2.0),
        "y_range": (0.0, 2.0),
        "device": "cpu",
        "dtype": "torch.float32",
        "least_count": 1.0,
        "init_value": 10.0,
        "alpha": 0.25,
    }
    base_kwargs.update(kwargs)
    return EMACostMapCfg(**base_kwargs)


def test_ema_cost_map_updates_current_cell_every_step():
    cost_map = EMACostMap(num_envs=1, cfg=make_cfg())
    current_pos = torch.tensor([[0.2, 0.2]])

    first_mask = cost_map.update(current_pos=current_pos, new_values=torch.tensor([18.0]))
    second_mask = cost_map.update(current_pos=current_pos, new_values=torch.tensor([26.0]))

    assert torch.equal(first_mask, torch.tensor([True]))
    assert torch.equal(second_mask, torch.tensor([True]))
    assert cost_map.maps[0, 0, 0, 0].item() == pytest.approx(15.5)


def test_ema_cost_map_updates_each_env_independently():
    cost_map = EMACostMap(num_envs=2, cfg=make_cfg(alpha=0.5, init_value=0.0))
    current_pos = torch.tensor([[0.2, 0.2], [1.2, 1.2]])
    new_values = torch.tensor([4.0, 10.0])

    cost_map.update(current_pos=current_pos, new_values=new_values)

    assert cost_map.maps[0, 0, 0, 0].item() == pytest.approx(2.0)
    assert cost_map.maps[1, 1, 1, 0].item() == pytest.approx(5.0)
    assert cost_map.maps[0, 1, 1, 0].item() == pytest.approx(0.0)
    assert cost_map.maps[1, 0, 0, 0].item() == pytest.approx(0.0)


def test_ema_cost_map_reset_clears_selected_env_map_contents():
    cost_map = EMACostMap(num_envs=2, cfg=make_cfg(alpha=1.0, init_value=3.0))
    current_pos = torch.tensor([[0.2, 0.2], [1.2, 1.2]])

    cost_map.update(current_pos=current_pos, new_values=torch.tensor([7.0, 11.0]))

    cost_map.reset(torch.tensor([1, 0]))

    assert torch.all(cost_map.maps[0] == 3.0)
    assert cost_map.maps[1, 1, 1, 0].item() == pytest.approx(11.0)


def test_ema_cost_map_load_is_optional_and_preserves_default_initialization():
    cost_map = EMACostMap(num_envs=1, cfg=make_cfg(init_value=3.0))

    assert torch.all(cost_map.maps == 3.0)


def test_ema_cost_map_load_restores_saved_map(tmp_path):
    source = EMACostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=3.0))
    source.update(current_pos=torch.tensor([[1.2, 0.2]]), new_values=torch.tensor([7.0]))
    source.save(tmp_path)

    target = EMACostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=3.0))
    target.load(tmp_path)

    assert torch.equal(target.maps, source.maps)


def test_ema_cost_map_load_uses_first_env_when_loading_single_env(tmp_path):
    source = EMACostMap(num_envs=5, cfg=make_cfg(alpha=1.0, init_value=3.0))
    source.maps.copy_(torch.arange(source.maps.numel(), dtype=torch.float32).view(source.maps.shape))
    source.save(tmp_path)

    target = EMACostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=3.0))
    target.load(tmp_path)

    assert torch.equal(target.maps, source.maps[:1])


def test_ema_cost_map_load_rejects_shape_mismatch(tmp_path):
    torch.save(torch.zeros(1, 3, 3, 1), tmp_path / "cost_map.pt")
    cost_map = EMACostMap(num_envs=1, cfg=make_cfg())

    with pytest.raises(ValueError, match="expected \\(1, 2, 2, 1\\)"):
        cost_map.load(tmp_path)


def test_ema_cost_map_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="alpha must be in \\(0, 1\\]"):
        EMACostMap(num_envs=1, cfg=make_cfg(alpha=0.0))


def test_ema_cost_map_exposes_planner_and_traversal_interfaces():
    cost_map = EMACostMap(num_envs=1, cfg=make_cfg(alpha=1.0, init_value=3.0))
    current_pos = torch.tensor([[1.2, 0.2]])

    cost_map.update(current_pos=current_pos, new_values=torch.tensor([7.0]))

    planner_map = cost_map.get_planner_map(env_id=0)

    assert planner_map.shape == (2, 2, 1)
    assert planner_map[1, 0, 0].item() == pytest.approx(7.0)
    assert cost_map.get_traversal_cost((0, 0), (1, 0), env_id=0) == pytest.approx(7.0)
