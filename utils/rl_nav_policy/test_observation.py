from types import SimpleNamespace

import torch

from utils.rl_nav_policy.observation import RLNavObservationBuilder, RLNavObservationConfig
from utils.diffusion_policy.dataset import crop_directional_map, crop_uncertainty_map, scale_uncertainty_crop


class FakeMap:
    def __init__(self, num_envs=1):
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            x_range=[-1.0, 3.0],
            y_range=[-1.0, 3.0],
            least_count=1.0,
            long_latent_dim=1,
            short_latent_dim=1,
            initial_uncertainty=100000.0,
        )
        self.maps = torch.arange(self.num_envs * 4 * 4 * 4 * 2, dtype=torch.float32).view(self.num_envs, 4, 4, 4, 2)
        self.uncertainties = torch.ones((self.num_envs, 4, 4, 4), dtype=torch.float32)


class FakeCellwiseMap:
    def __init__(self, num_envs=1):
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            x_range=[-1.0, 3.0],
            y_range=[-1.0, 3.0],
            least_count=1.0,
            long_latent_dim=1,
            short_latent_dim=1,
            initial_uncertainty=100000.0,
        )
        self.maps = torch.arange(self.num_envs * 4 * 4 * 2, dtype=torch.float32).view(self.num_envs, 4, 4, 2)
        self.uncertainties = torch.ones((self.num_envs, 4, 4), dtype=torch.float32)


class FakeEnv:
    def __init__(self, num_envs=1):
        self.num_envs = int(num_envs)
        self.device = torch.device("cpu")
        self.root_local_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32)
        self.root_states = torch.zeros((self.num_envs, 13), dtype=torch.float32)
        self.root_states[:, 3:7] = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(self.num_envs, 1)
        self.commands = torch.zeros((self.num_envs, 4), dtype=torch.float32)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32)


def test_rl_nav_observation_layout_and_goal_geometry():
    cfg = RLNavObservationConfig(crop_size=2, include_previous_command=True)
    builder = RLNavObservationBuilder(cfg, device="cpu")
    env = FakeEnv()
    neural_map = FakeMap()

    obs = builder.build_observation(
        env,
        neural_map,
        torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        previous_command=torch.zeros((1, 3)),
    )

    assert obs.shape == (1, builder.observation_dim(neural_map))
    assert builder.token_dim(neural_map) == 18
    assert builder.condition_dim == 14
    condition = obs[:, -builder.condition_dim :]
    assert torch.allclose(condition[0, :3], torch.tensor([1.0, 0.0, 2.0 / 12.0]))


def test_rl_nav_condition_observation_matches_full_condition_tail():
    cfg = RLNavObservationConfig(crop_size=2, include_previous_command=True)
    builder = RLNavObservationBuilder(cfg, device="cpu")
    env = FakeEnv()
    neural_map = FakeMap()
    goal_xy = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    previous_command = torch.zeros((1, 3), dtype=torch.float32)

    full_obs = builder.build_observation(env, neural_map, goal_xy, previous_command=previous_command)
    condition_obs = builder.build_condition_observation(env, goal_xy, previous_command=previous_command)

    assert condition_obs.shape == (1, builder.condition_observation_dim())
    assert torch.allclose(condition_obs, full_obs[:, -builder.condition_dim :])


def test_rl_nav_observation_v3_dual_scale_layout():
    cfg = RLNavObservationConfig(crop_size=2, include_global_map=True, global_pool_size=2, include_previous_command=True)
    builder = RLNavObservationBuilder(cfg, device="cpu")
    env = FakeEnv()
    neural_map = FakeMap()

    batch = builder.build_batch(
        env,
        neural_map,
        torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        previous_command=torch.zeros((1, 3)),
    )
    obs = builder.build_observation(
        env,
        neural_map,
        torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        previous_command=torch.zeros((1, 3)),
    )

    assert builder.num_tokens() == 4
    assert builder.global_num_tokens() == 4
    assert batch["map_tokens"].shape == (1, 4, builder.token_dim(neural_map))
    assert batch["global_map_tokens"].shape == (1, 4, builder.token_dim(neural_map))
    assert obs.shape == (1, builder.observation_dim(neural_map))
    assert torch.all(torch.isfinite(batch["global_map_tokens"]))


def test_rl_nav_observation_supports_cellwise_map_layout():
    cfg = RLNavObservationConfig(crop_size=2, include_previous_command=True)
    builder = RLNavObservationBuilder(cfg, device="cpu")
    env = FakeEnv()
    neural_map = FakeCellwiseMap()

    obs = builder.build_observation(
        env,
        neural_map,
        torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        previous_command=torch.zeros((1, 3)),
    )

    assert obs.shape == (1, builder.observation_dim(neural_map))
    assert builder.token_dim(neural_map) == 9
    assert builder.condition_dim == 14
    tokens = obs[:, : builder.num_tokens() * builder.token_dim(neural_map)].view(
        1,
        builder.num_tokens(),
        builder.token_dim(neural_map),
    )
    assert tokens.shape == (1, 4, 9)


def test_batched_map_tokens_match_single_env_reference():
    cfg = RLNavObservationConfig(crop_size=3, include_previous_command=True)
    builder = RLNavObservationBuilder(cfg, device="cpu")
    env = FakeEnv(num_envs=2)
    env.root_local_pos[:, :2] = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    neural_map = FakeMap(num_envs=2)
    goal_xy = torch.tensor([[2.0, 0.0], [2.0, 2.0]], dtype=torch.float32)

    batch = builder.build_batch(env, neural_map, goal_xy, previous_command=torch.zeros((2, 3)))
    start_grid = batch["start_grid"]

    for env_id in range(2):
        map_crop = crop_directional_map(
            neural_map.maps[env_id],
            start_grid[env_id],
            crop_size=cfg.crop_size,
            pad_value=0.0,
        )
        uncertainty_crop = crop_uncertainty_map(
            neural_map.uncertainties[env_id],
            start_grid[env_id],
            crop_size=cfg.crop_size,
            pad_value=cfg.uncertainty_max,
        )
        scaled_uncertainty = scale_uncertainty_crop(
            uncertainty_crop,
            uncertainty_log_scale=cfg.uncertainty_log_scale,
        )
        cell_xy = builder.crop_cell_xy(neural_map, start_grid[env_id], "cpu")
        expected = builder.build_map_tokens(
            map_crop,
            scaled_uncertainty,
            cell_xy,
            env.root_local_pos[env_id, :2],
            goal_xy[env_id],
        )
        assert torch.allclose(batch["map_tokens"][env_id], expected)
