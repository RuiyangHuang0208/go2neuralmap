from types import SimpleNamespace

import torch

from utils.rl_nav_policy.env import RLNavEnvConfig, RLNavPolicyEnv, RLNavTerrainResamplingConfig


class FakeResetEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 1
        self.root_local_pos = torch.zeros((1, 3), dtype=torch.float32)
        self.commands = torch.zeros((1, 4), dtype=torch.float32)
        self.task_start_xy = torch.zeros((1, 2), dtype=torch.float32)
        self.task_start_valid = torch.zeros(1, dtype=torch.bool)
        self.task_goal_xy = torch.zeros((1, 2), dtype=torch.float32)
        self.task_goal_valid = torch.zeros(1, dtype=torch.bool)
        self.terrain_reset_calls = 0
        self.task_reset_calls = 0

    def terrain_reset(self, env_ids, *, resample_goal):
        self.terrain_reset_calls += 1
        return SimpleNamespace(start_xy=torch.tensor([[3.0, 4.0]], dtype=torch.float32))

    def task_reset(self, env_ids, *, resample_goal):
        self.task_reset_calls += 1
        return SimpleNamespace(start_xy=torch.tensor([[1.0, 2.0]], dtype=torch.float32))

    def compute_observations(self):
        pass


def make_wrapper(resampling_cfg):
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 1
    wrapper.env = FakeResetEnv()
    wrapper.cfg = RLNavEnvConfig(terrain_resampling=resampling_cfg)
    wrapper.task_start_xy = torch.zeros((1, 2), dtype=torch.float32)
    wrapper.current_command = torch.ones((1, 3), dtype=torch.float32)
    wrapper.previous_command = torch.ones((1, 3), dtype=torch.float32)
    wrapper.episode_length_buf = torch.ones(1, dtype=torch.long)
    wrapper.collision_count = torch.ones(1, dtype=torch.long)
    wrapper.success_count_since_map_reset = torch.ones(1, dtype=torch.long)
    wrapper.goal_xy = torch.zeros((1, 2), dtype=torch.float32)
    wrapper._refresh_root_local_pos = lambda env_ids: None
    wrapper._store_current_start = lambda env_ids: None
    wrapper._sample_goals = lambda env_ids: None
    wrapper._reset_maps = lambda env_ids: None
    return wrapper


def test_reset_task_uses_terrain_reset_when_requested():
    wrapper = make_wrapper(RLNavTerrainResamplingConfig())

    wrapper._reset_task(torch.tensor([0]), clear_map=False, resample_terrain=True)

    assert wrapper.env.terrain_reset_calls == 1
    assert wrapper.env.task_reset_calls == 0
    assert torch.allclose(wrapper.task_start_xy[0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(wrapper.current_command, torch.zeros_like(wrapper.current_command))


def test_reset_task_uses_task_reset_by_default():
    wrapper = make_wrapper(RLNavTerrainResamplingConfig())

    wrapper._reset_task(torch.tensor([0]), clear_map=False)

    assert wrapper.env.terrain_reset_calls == 0
    assert wrapper.env.task_reset_calls == 1
    assert torch.allclose(wrapper.task_start_xy[0], torch.tensor([1.0, 2.0]))
