from types import SimpleNamespace

import torch

from utils.rl_nav_policy.env import RLNavEnvConfig, RLNavPolicyEnv
from utils.rl_nav_policy.observation import RLNavObservationBuilder, RLNavObservationConfig


class FakeMap:
    def __init__(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.cfg = SimpleNamespace(
            x_range=[-1.0, 3.0],
            y_range=[-1.0, 3.0],
            least_count=1.0,
            long_latent_dim=1,
            short_latent_dim=1,
        )
        self.maps = torch.zeros((2, 4, 4, 4, 2), dtype=torch.float32)
        self.uncertainties = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
        self.uncertainties[1] = 1.0e9


class FakeEnv:
    def __init__(self):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.root_local_pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )


class FakeMiddleTerrainLoader:
    def __init__(self):
        self.num_middle_points = torch.tensor([2, 0], dtype=torch.long)
        self.points = torch.tensor(
            [
                [[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

    def get_num_middle_points(self):
        return self.num_middle_points

    def sample_middle_points(self, terrain_ids):
        ids = terrain_ids.to(dtype=torch.long).view(-1)
        return self.points[ids, :1]


def test_collision_penalty_scale_uses_crop_uncertainty_confidence():
    cfg = RLNavEnvConfig(observation=RLNavObservationConfig(crop_size=2))
    cfg.reward.collision_uncertainty_scale.enabled = True
    cfg.reward.collision_uncertainty_scale.min_scale = 0.5
    cfg.reward.collision_uncertainty_scale.max_scale = 2.0

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg
    wrapper.env = FakeEnv()
    wrapper.neural_map = FakeMap()
    wrapper.obs_builder = RLNavObservationBuilder(cfg.observation, device="cpu")

    scale = wrapper._collision_penalty_scale()

    assert torch.allclose(scale, torch.tensor([2.0, 0.5]))


def test_collision_penalty_scale_disabled_returns_one():
    cfg = RLNavEnvConfig()
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg

    assert torch.allclose(wrapper._collision_penalty_scale(), torch.ones(2))


def test_middle_point_reward_is_added_when_reached():
    cfg = RLNavEnvConfig()
    cfg.reward.progress_scale = 0.0
    cfg.reward.time_penalty = 0.0
    cfg.reward.command_penalty_scale = 0.0
    cfg.reward.command_smoothness_penalty_scale = 0.0
    cfg.reward.middle_point_reward = 3.0

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg

    reward = wrapper._compute_reward(
        command=torch.zeros((2, 3)),
        previous_command=torch.zeros((2, 3)),
        previous_distance=torch.zeros(2),
        current_distance=torch.zeros(2),
        reached_goal=torch.zeros(2, dtype=torch.bool),
        reached_middle=torch.tensor([True, False]),
        collision=torch.zeros(2, dtype=torch.bool),
        fixed_collision=torch.zeros(2, dtype=torch.bool),
        collision_penalty_scale=torch.ones(2),
        repeated_collision_penalty=torch.zeros(2),
        terminated=torch.zeros(2, dtype=torch.bool),
    )

    assert torch.allclose(reward, torch.tensor([3.0, 0.0]))


def test_middle_point_status_only_reaches_unclaimed_valid_points():
    cfg = RLNavEnvConfig()
    cfg.reward.middle_point_tolerance = 0.35

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg
    wrapper.env = FakeEnv()
    wrapper.middle_xy = torch.tensor([[0.2, 0.0], [0.2, 0.0]], dtype=torch.float32)
    wrapper.middle_valid = torch.tensor([True, True])
    wrapper.middle_reward_claimed = torch.tensor([False, True])

    _, distance, reached = wrapper._middle_point_status()

    assert torch.allclose(distance, torch.tensor([0.2, 0.2]))
    assert torch.equal(reached, torch.tensor([True, False]))


def test_sample_middle_points_resets_claimed_state_per_task():
    cfg = RLNavEnvConfig()

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg
    wrapper.middle_xy = torch.ones((2, 2), dtype=torch.float32)
    wrapper.middle_valid = torch.ones(2, dtype=torch.bool)
    wrapper.middle_reward_claimed = torch.ones(2, dtype=torch.bool)
    wrapper.env = FakeEnv()
    wrapper.env.env_terrain_map = torch.tensor([0, 1], dtype=torch.long)
    wrapper.env.terrain_loader = FakeMiddleTerrainLoader()

    wrapper._sample_middle_points(torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(wrapper.middle_xy[0], torch.tensor([0.5, 0.0]))
    assert torch.allclose(wrapper.middle_xy[1], torch.tensor([0.0, 0.0]))
    assert torch.equal(wrapper.middle_valid, torch.tensor([True, False]))
    assert torch.equal(wrapper.middle_reward_claimed, torch.tensor([False, False]))


def test_dynamic_collision_penalty_does_not_scale_fixed_collision():
    cfg = RLNavEnvConfig()
    cfg.reward.progress_scale = 0.0
    cfg.reward.time_penalty = 0.0
    cfg.reward.command_penalty_scale = 0.0
    cfg.reward.command_smoothness_penalty_scale = 0.0
    cfg.reward.collision_penalty = 2.0

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg

    reward = wrapper._compute_reward(
        command=torch.zeros((2, 3)),
        previous_command=torch.zeros((2, 3)),
        previous_distance=torch.zeros(2),
        current_distance=torch.zeros(2),
        reached_goal=torch.zeros(2, dtype=torch.bool),
        collision=torch.tensor([True, False]),
        fixed_collision=torch.tensor([False, True]),
        collision_penalty_scale=torch.tensor([2.0, 2.0]),
        repeated_collision_penalty=torch.zeros(2),
        terminated=torch.zeros(2, dtype=torch.bool),
    )

    assert torch.allclose(reward, torch.tensor([-4.0, -2.0]))


def test_repeated_collision_penalty_accumulates_at_same_location():
    cfg = RLNavEnvConfig()
    cfg.reward.repeated_collision_penalty.enabled = True
    cfg.reward.repeated_collision_penalty.penalty_scale = 3.0
    cfg.reward.repeated_collision_penalty.radius_cells = 0
    cfg.reward.repeated_collision_penalty.max_count = 3.0

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg
    wrapper.env = FakeEnv()
    wrapper.neural_map = FakeMap()
    wrapper.repeated_collision_map = wrapper._new_repeated_collision_map()

    first = wrapper._repeated_collision_penalty(torch.tensor([True, False]))
    second = wrapper._repeated_collision_penalty(torch.tensor([True, False]))

    assert torch.allclose(first, torch.zeros(2))
    assert torch.allclose(second, torch.tensor([1.0, 0.0]))


def test_repeated_collision_penalty_uses_neighbor_radius():
    cfg = RLNavEnvConfig()
    cfg.reward.repeated_collision_penalty.enabled = True
    cfg.reward.repeated_collision_penalty.penalty_scale = 2.0
    cfg.reward.repeated_collision_penalty.radius_cells = 1
    cfg.reward.repeated_collision_penalty.max_count = 2.0

    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg
    wrapper.env = FakeEnv()
    wrapper.neural_map = FakeMap()
    wrapper.repeated_collision_map = wrapper._new_repeated_collision_map()

    wrapper.env.root_local_pos[0, :2] = torch.tensor([0.0, 0.0])
    first = wrapper._repeated_collision_penalty(torch.tensor([True, False]))
    wrapper.env.root_local_pos[0, :2] = torch.tensor([1.0, 0.0])
    second = wrapper._repeated_collision_penalty(torch.tensor([True, False]))

    assert torch.allclose(first, torch.zeros(2))
    assert torch.allclose(second, torch.tensor([1.0, 0.0]))


def test_episode_reset_keeps_repeated_collision_map_but_map_reset_clears_it():
    cfg = RLNavEnvConfig()
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.cfg = cfg
    wrapper.env = FakeEnv()
    wrapper.neural_map = FakeMap()
    wrapper.episode_length_buf = torch.ones(2, dtype=torch.long)
    wrapper.collision_count = torch.ones(2, dtype=torch.long)
    wrapper.success_count_since_map_reset = torch.ones(2, dtype=torch.long)
    wrapper.repeated_collision_map = torch.ones((2, 4, 4), dtype=torch.float32)

    wrapper._reset_episode_buffers(torch.tensor([0]))
    assert torch.allclose(wrapper.repeated_collision_map[0], torch.ones((4, 4)))

    def fake_reset_task(env_ids, *, clear_map, resample_terrain=False):
        assert clear_map is True

    wrapper._reset_task = fake_reset_task
    wrapper._reset_map_episode(torch.tensor([0]))

    assert torch.allclose(wrapper.repeated_collision_map[0], torch.zeros((4, 4)))
    assert torch.allclose(wrapper.repeated_collision_map[1], torch.ones((4, 4)))
