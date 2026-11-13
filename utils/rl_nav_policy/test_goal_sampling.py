from types import SimpleNamespace

import torch

from utils.rl_nav_policy.env import RLNavEnvConfig, RLNavGoalSamplingConfig, RLNavPolicyEnv


class SequenceTerrainLoader:
    def __init__(self, samples):
        self.samples = [torch.tensor(sample, dtype=torch.float32) for sample in samples]
        self.index = 0

    def sample_goals(self, terrain_ids):
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return sample[: terrain_ids.numel()].view(terrain_ids.numel(), 1, 3)


def make_env(samples):
    env = SimpleNamespace()
    env.device = torch.device("cpu")
    env.num_envs = 2
    env.root_local_pos = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], dtype=torch.float32)
    env.env_terrain_map = torch.tensor([0, 1], dtype=torch.long)
    env.terrain_loader = SequenceTerrainLoader(samples)
    env.task_goal_xy = torch.zeros((2, 2), dtype=torch.float32)
    env.task_goal_valid = torch.zeros(2, dtype=torch.bool)
    return env


def test_goal_sampling_filters_by_distance_and_syncs_env_state():
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.env = make_env(
        [
            [[0.2, 0.0, 0.0], [9.0, 5.0, 0.0]],
            [[2.0, 0.0, 0.0], [7.0, 5.0, 0.0]],
        ]
    )
    wrapper.cfg = RLNavEnvConfig(
        goal_sampling=RLNavGoalSamplingConfig(
            min_distance=1.0,
            max_distance=3.0,
            max_attempts=2,
            fallback="accept_last",
        )
    )
    wrapper.goal_xy = torch.zeros((2, 2), dtype=torch.float32)

    wrapper._sample_goals(torch.tensor([0, 1], dtype=torch.long))

    expected = torch.tensor([[2.0, 0.0], [7.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(wrapper.goal_xy, expected)
    assert torch.allclose(wrapper.env.task_goal_xy, expected)
    assert torch.equal(wrapper.env.task_goal_valid, torch.tensor([True, True]))


def test_goal_sampling_accept_last_fallback_after_failed_filter():
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.env = make_env(
        [
            [[0.2, 0.0, 0.0], [9.0, 5.0, 0.0]],
            [[0.3, 0.0, 0.0], [9.5, 5.0, 0.0]],
        ]
    )
    wrapper.cfg = RLNavEnvConfig(
        goal_sampling=RLNavGoalSamplingConfig(
            min_distance=1.0,
            max_distance=3.0,
            max_attempts=2,
            fallback="accept_last",
        )
    )
    wrapper.goal_xy = torch.zeros((2, 2), dtype=torch.float32)

    wrapper._sample_goals(torch.tensor([0, 1], dtype=torch.long))

    expected = torch.tensor([[0.3, 0.0], [9.5, 5.0]], dtype=torch.float32)
    assert torch.allclose(wrapper.goal_xy, expected)


def test_goal_sampling_can_avoid_previous_goal():
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.env = make_env(
        [
            [[1.0, 0.0, 0.0], [6.0, 5.0, 0.0]],
            [[2.0, 0.0, 0.0], [7.0, 5.0, 0.0]],
        ]
    )
    wrapper.cfg = RLNavEnvConfig(
        goal_sampling=RLNavGoalSamplingConfig(
            max_attempts=2,
            fallback="accept_last",
            avoid_previous=True,
            previous_tolerance=1.0e-4,
        )
    )
    wrapper.goal_xy = torch.tensor([[1.0, 0.0], [6.0, 5.0]], dtype=torch.float32)
    wrapper.goal_initialized = torch.tensor([True, True])

    wrapper._sample_goals(torch.tensor([0, 1], dtype=torch.long))

    expected = torch.tensor([[2.0, 0.0], [7.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(wrapper.goal_xy, expected)
    assert torch.allclose(wrapper.env.task_goal_xy, expected)


def test_goal_sampling_avoid_previous_does_not_filter_uninitialized_goals():
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 2
    wrapper.env = make_env(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    wrapper.cfg = RLNavEnvConfig(
        goal_sampling=RLNavGoalSamplingConfig(
            max_attempts=1,
            fallback="accept_last",
            avoid_previous=True,
        )
    )
    wrapper.goal_xy = torch.zeros((2, 2), dtype=torch.float32)
    wrapper.goal_initialized = torch.tensor([False, False])

    wrapper._sample_goals(torch.tensor([0, 1], dtype=torch.long))

    assert torch.allclose(wrapper.goal_xy, torch.zeros((2, 2), dtype=torch.float32))
    assert torch.equal(wrapper.goal_initialized, torch.tensor([True, True]))
