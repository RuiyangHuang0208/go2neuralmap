import pytest
import torch

from utils.neural_map import (
    DirectionalBayesianNeuralMap,
    DirectionalBayesianNeuralMapCfg,
    EMANeuralMap,
    EMANeuralMapCfg,
)


def make_ema_cfg(**kwargs) -> EMANeuralMapCfg:
    base_kwargs = {
        "x_range": (0.0, 2.0),
        "y_range": (0.0, 2.0),
        "device": "cpu",
        "dtype": "torch.float32",
        "least_count": 1.0,
        "long_latent_dim": 2,
        "short_latent_dim": 1,
    }
    base_kwargs.update(kwargs)
    return EMANeuralMapCfg(**base_kwargs)


def make_directional_bayesian_cfg(**kwargs) -> DirectionalBayesianNeuralMapCfg:
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
    return DirectionalBayesianNeuralMapCfg(**base_kwargs)


def test_ema_neural_map_load_is_optional_and_preserves_default_initialization():
    neural_map = EMANeuralMap(num_envs=1, cfg=make_ema_cfg())

    assert torch.all(neural_map.long_maps == 0.0)
    assert torch.all(neural_map.short_maps == 0.0)


def test_ema_neural_map_load_restores_saved_maps_and_labels(tmp_path):
    cfg = make_ema_cfg(collect_true_terrain_label=True, unknown_label=-1)
    source = EMANeuralMap(num_envs=1, cfg=cfg)
    source.regist_true_terrain_label(torch.tensor([4]))
    source.update(
        long_latents=torch.tensor([[1.0, 2.0]]),
        short_latents=torch.tensor([[3.0]]),
        xy_coord=torch.tensor([[1.2, 0.2]]),
    )
    source.save(tmp_path)

    target = EMANeuralMap(num_envs=1, cfg=cfg)
    target.load(tmp_path)

    assert torch.equal(target.long_maps, source.long_maps)
    assert torch.equal(target.short_maps, source.short_maps)
    assert torch.equal(target.terrain_labels, source.terrain_labels)


def test_ema_neural_map_load_rejects_shape_mismatch(tmp_path):
    torch.save(torch.zeros(1, 3, 2, 2), tmp_path / "long_map.pt")
    torch.save(torch.zeros(1, 2, 2, 1), tmp_path / "short_map.pt")
    neural_map = EMANeuralMap(num_envs=1, cfg=make_ema_cfg())

    with pytest.raises(ValueError, match="expected \\(1, 2, 2, 2\\)"):
        neural_map.load(tmp_path)


def test_directional_bayesian_neural_map_load_restores_full_state(tmp_path):
    cfg = make_directional_bayesian_cfg()
    source = DirectionalBayesianNeuralMap(num_envs=1, cfg=cfg)
    source.update(
        long_latents=torch.tensor([[1.0, 2.0]]),
        short_latents=torch.tensor([[3.0]]),
        xy_coord=torch.tensor([[0.2, 0.2]]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
    )
    source.update(
        long_latents=torch.tensor([[4.0, 5.0]]),
        short_latents=torch.tensor([[6.0]]),
        xy_coord=torch.tensor([[1.2, 1.2]]),
        root_velocity=torch.tensor([[0.0, 1.0]]),
    )
    source.save(tmp_path)

    target = DirectionalBayesianNeuralMap(num_envs=1, cfg=cfg)
    target.load(tmp_path)

    assert torch.equal(target.maps, source.maps)
    assert torch.equal(target.long_maps, source.long_maps)
    assert torch.equal(target.short_maps, source.short_maps)
    assert torch.equal(target.uncertainties, source.uncertainties)
    assert torch.equal(target.last_update_steps, source.last_update_steps)
    assert torch.equal(target.current_step, source.current_step)


def test_directional_bayesian_neural_map_load_uses_first_env_when_loading_single_env(tmp_path):
    cfg = make_directional_bayesian_cfg()
    source = DirectionalBayesianNeuralMap(num_envs=5, cfg=cfg)
    source.maps.copy_(torch.arange(source.maps.numel(), dtype=torch.float32).view(source.maps.shape))
    source.uncertainties.copy_(
        torch.arange(source.uncertainties.numel(), dtype=torch.float32).view(source.uncertainties.shape)
    )
    source.last_update_steps.copy_(
        torch.arange(source.last_update_steps.numel(), dtype=torch.long).view(source.last_update_steps.shape)
    )
    source.current_step.fill_(7)
    source.save(tmp_path)

    target = DirectionalBayesianNeuralMap(num_envs=1, cfg=cfg)
    target.load(tmp_path)

    assert torch.equal(target.maps, source.maps[:1])
    assert torch.equal(target.long_maps, source.long_maps[:1])
    assert torch.equal(target.short_maps, source.short_maps[:1])
    assert torch.equal(target.uncertainties, source.uncertainties[:1])
    assert torch.equal(target.last_update_steps, source.last_update_steps[:1])
    assert torch.equal(target.current_step, source.current_step)


def test_directional_bayesian_neural_map_reset_clears_selected_env_only():
    cfg = make_directional_bayesian_cfg(initial_uncertainty=10.0, max_uncertainty=20.0)
    neural_map = DirectionalBayesianNeuralMap(num_envs=2, cfg=cfg)
    neural_map.update(
        long_latents=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        short_latents=torch.tensor([[5.0], [6.0]]),
        xy_coord=torch.tensor([[0.2, 0.2], [1.2, 1.2]]),
        root_velocity=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    neural_map.update(
        long_latents=torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        short_latents=torch.tensor([[11.0], [12.0]]),
        xy_coord=torch.tensor([[0.2, 0.2], [1.2, 1.2]]),
        root_velocity=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
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
    assert torch.equal(neural_map.current_step, current_step)


def test_directional_bayesian_neural_map_get_aged_uncertainties_is_read_only():
    cfg = make_directional_bayesian_cfg(initial_uncertainty=10.0, max_uncertainty=10.0, aging_rate=0.5)
    neural_map = DirectionalBayesianNeuralMap(num_envs=1, cfg=cfg)
    neural_map.update(
        long_latents=torch.tensor([[1.0, 2.0]]),
        short_latents=torch.tensor([[3.0]]),
        xy_coord=torch.tensor([[0.2, 0.2]]),
        root_velocity=torch.tensor([[1.0, 0.0]]),
    )
    stored_uncertainties = neural_map.uncertainties.clone()
    neural_map.update(
        long_latents=torch.tensor([[4.0, 5.0]]),
        short_latents=torch.tensor([[6.0]]),
        xy_coord=torch.tensor([[1.2, 1.2]]),
        root_velocity=torch.tensor([[0.0, 1.0]]),
    )

    aged = neural_map.get_aged_uncertainties()

    assert aged[0, 0, 0, DirectionalBayesianNeuralMap.POS_X] > stored_uncertainties[
        0, 0, 0, DirectionalBayesianNeuralMap.POS_X
    ]
    assert aged.max() <= float(cfg.max_uncertainty)
    assert torch.equal(neural_map.uncertainties[0, 0, 0], stored_uncertainties[0, 0, 0])
