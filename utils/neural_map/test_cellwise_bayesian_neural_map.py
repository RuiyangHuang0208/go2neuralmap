import pytest
import torch

from utils.neural_map import CellwiseBayesianNeuralMap, CellwiseBayesianNeuralMapCfg


def make_cellwise_bayesian_cfg(**kwargs) -> CellwiseBayesianNeuralMapCfg:
    base_kwargs = {
        "x_range": (0.0, 2.0),
        "y_range": (0.0, 2.0),
        "device": "cpu",
        "dtype": "torch.float32",
        "least_count": 1.0,
        "long_latent_dim": 2,
        "short_latent_dim": 1,
        "initial_uncertainty": 10.0,
        "incoming_uncertainty": 2.0,
        "aging_rate": 0.1,
        "max_uncertainty": 10.0,
    }
    base_kwargs.update(kwargs)
    return CellwiseBayesianNeuralMapCfg(**base_kwargs)


def test_cellwise_bayesian_neural_map_shapes():
    neural_map = CellwiseBayesianNeuralMap(num_envs=2, cfg=make_cellwise_bayesian_cfg())

    assert neural_map.maps.shape == (2, 2, 2, 3)
    assert neural_map.long_maps.shape == (2, 2, 2, 2)
    assert neural_map.short_maps.shape == (2, 2, 2, 1)
    assert neural_map.uncertainties.shape == (2, 2, 2)
    assert neural_map.last_update_steps.shape == (2, 2, 2)


def test_cellwise_bayesian_neural_map_update_writes_current_cell():
    cfg = make_cellwise_bayesian_cfg(initial_uncertainty=10.0, incoming_uncertainty=2.0)
    neural_map = CellwiseBayesianNeuralMap(num_envs=1, cfg=cfg)

    neural_map.update(
        long_latents=torch.tensor([[1.0, 2.0]]),
        short_latents=torch.tensor([[3.0]]),
        xy_coord=torch.tensor([[0.2, 0.2]]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
    )

    expected_gain = float(cfg.initial_uncertainty) / (
        float(cfg.initial_uncertainty) + float(cfg.incoming_uncertainty)
    )
    assert torch.allclose(neural_map.maps[0, 0, 0], torch.tensor([1.0, 2.0, 3.0]) * expected_gain)
    assert neural_map.uncertainties[0, 0, 0].item() == pytest.approx((1.0 - expected_gain) * 10.0)
    assert neural_map.current_step.item() == 1


def test_cellwise_bayesian_neural_map_load_restores_full_state(tmp_path):
    cfg = make_cellwise_bayesian_cfg()
    source = CellwiseBayesianNeuralMap(num_envs=1, cfg=cfg)
    source.update(
        long_latents=torch.tensor([[1.0, 2.0]]),
        short_latents=torch.tensor([[3.0]]),
        xy_coord=torch.tensor([[0.2, 0.2]]),
    )
    source.update(
        long_latents=torch.tensor([[4.0, 5.0]]),
        short_latents=torch.tensor([[6.0]]),
        xy_coord=torch.tensor([[1.2, 1.2]]),
    )
    source.save(tmp_path)

    target = CellwiseBayesianNeuralMap(num_envs=1, cfg=cfg)
    target.load(tmp_path)

    assert torch.equal(target.maps, source.maps)
    assert torch.equal(target.long_maps, source.long_maps)
    assert torch.equal(target.short_maps, source.short_maps)
    assert torch.equal(target.uncertainties, source.uncertainties)
    assert torch.equal(target.last_update_steps, source.last_update_steps)
    assert torch.equal(target.current_step, source.current_step)


def test_cellwise_bayesian_neural_map_reset_clears_selected_env_only():
    cfg = make_cellwise_bayesian_cfg(initial_uncertainty=10.0, max_uncertainty=20.0)
    neural_map = CellwiseBayesianNeuralMap(num_envs=2, cfg=cfg)
    neural_map.update(
        long_latents=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        short_latents=torch.tensor([[5.0], [6.0]]),
        xy_coord=torch.tensor([[0.2, 0.2], [1.2, 1.2]]),
    )
    current_step = neural_map.current_step.clone()
    env_1_maps = neural_map.maps[1].clone()
    env_1_uncertainties = neural_map.uncertainties[1].clone()
    env_1_last_update_steps = neural_map.last_update_steps[1].clone()

    neural_map.reset(torch.tensor([True, False]))

    assert torch.all(neural_map.maps[0] == 0.0)
    assert torch.all(neural_map.uncertainties[0] == float(cfg.initial_uncertainty))
    assert torch.all(neural_map.last_update_steps[0] == current_step)
    assert torch.equal(neural_map.maps[1], env_1_maps)
    assert torch.equal(neural_map.uncertainties[1], env_1_uncertainties)
    assert torch.equal(neural_map.last_update_steps[1], env_1_last_update_steps)


def test_cellwise_bayesian_neural_map_get_aged_uncertainties_is_read_only():
    cfg = make_cellwise_bayesian_cfg(initial_uncertainty=10.0, max_uncertainty=10.0, aging_rate=0.5)
    neural_map = CellwiseBayesianNeuralMap(num_envs=1, cfg=cfg)
    neural_map.update(
        long_latents=torch.tensor([[1.0, 2.0]]),
        short_latents=torch.tensor([[3.0]]),
        xy_coord=torch.tensor([[0.2, 0.2]]),
    )
    stored_uncertainties = neural_map.uncertainties.clone()
    neural_map.update(
        long_latents=torch.tensor([[4.0, 5.0]]),
        short_latents=torch.tensor([[6.0]]),
        xy_coord=torch.tensor([[1.2, 1.2]]),
    )

    aged = neural_map.get_aged_uncertainties()

    assert aged[0, 0, 0] > stored_uncertainties[0, 0, 0]
    assert aged.max() <= float(cfg.max_uncertainty)
    assert torch.equal(neural_map.uncertainties[0, 0, 0], stored_uncertainties[0, 0, 0])
