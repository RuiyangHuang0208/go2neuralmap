from types import SimpleNamespace

import torch

from utils.residual_policy.observation import ResidualObservationBuilder, ResidualObservationConfig


class FakeEncoder:
    def __call__(self, batch):
        return torch.ones(batch["map_crop"].shape[0], 7, device=batch["map_crop"].device)


class FakeNeuralMapWithAging(SimpleNamespace):
    def get_aged_uncertainties(self):
        return self.aged_uncertainties


def test_residual_observation_builder_shapes():
    cfg = ResidualObservationConfig(crop_size=3, model_horizon=5, command_dim=4, goal_scale=10.0)
    builder = ResidualObservationBuilder(cfg, device="cpu")
    neural_map = SimpleNamespace(
        cfg=SimpleNamespace(x_range=(-1.0, 4.0), y_range=(-1.0, 4.0), least_count=1.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        maps=torch.zeros(2, 5, 5, 4, 3),
        uncertainties=torch.ones(2, 5, 5, 4),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        root_local_pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]]),
        root_states=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    goal_xy = torch.tensor([[1.0, 0.0], [1.0, 3.0]])

    batch = builder.build_batch(env, neural_map, goal_xy)
    obs = builder.build_observation(env, neural_map, goal_xy, FakeEncoder(), batch=batch)

    assert batch["map_crop"].shape == (2, 12, 3, 3)
    assert batch["uncertainty_crop"].shape == (2, 4, 3, 3)
    assert batch["current_latent"].shape == (2, 12)
    assert batch["target"].shape == (2, 5, 4)
    assert obs.shape == (2, 7 + 12 + 3)
    assert torch.allclose(obs[:, -3:-1], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))


def test_residual_observation_builder_uses_aged_uncertainties():
    cfg = ResidualObservationConfig(crop_size=3, model_horizon=5, command_dim=4, goal_scale=10.0, uncertainty_log_scale=1.0)
    builder = ResidualObservationBuilder(cfg, device="cpu")
    neural_map = FakeNeuralMapWithAging(
        cfg=SimpleNamespace(x_range=(-1.0, 4.0), y_range=(-1.0, 4.0), least_count=1.0),
        device=torch.device("cpu"),
        dtype=torch.float32,
        maps=torch.zeros(1, 5, 5, 4, 3),
        uncertainties=torch.zeros(1, 5, 5, 4),
        aged_uncertainties=torch.ones(1, 5, 5, 4),
    )
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        root_local_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        root_states=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    )

    batch = builder.build_batch(env, neural_map, torch.tensor([[1.0, 0.0]]))

    expected = torch.log1p(torch.ones_like(batch["uncertainty_crop"]))
    assert torch.allclose(batch["uncertainty_crop"], expected)
