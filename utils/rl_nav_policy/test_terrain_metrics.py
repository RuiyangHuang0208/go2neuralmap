from types import SimpleNamespace

import torch

from utils.rl_nav_policy.env import RLNavPolicyEnv


def test_terrain_episode_metrics_groups_done_envs_by_terrain_id():
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 4
    wrapper.env = SimpleNamespace(env_terrain_map=torch.tensor([0, 1, 1, 0], dtype=torch.long))

    dones = torch.tensor([True, True, False, True])
    reached_goal = torch.tensor([True, False, True, False])

    metrics = wrapper._terrain_episode_metrics(dones, reached_goal)

    assert torch.allclose(metrics["terrain_0_success_rate"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(metrics["terrain_0_episode_count"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(metrics["terrain_1_success_rate"], torch.tensor([0.0]))
    assert torch.allclose(metrics["terrain_1_episode_count"], torch.tensor([1.0]))


def test_terrain_episode_metrics_include_sanitized_terrain_names():
    wrapper = RLNavPolicyEnv.__new__(RLNavPolicyEnv)
    wrapper.device = torch.device("cpu")
    wrapper.num_envs = 4
    terrain_loader = SimpleNamespace(
        terrain_instances=[
            SimpleNamespace(name="d1-plane"),
            SimpleNamespace(name="d1 block/forest"),
        ],
    )
    wrapper.env = SimpleNamespace(
        env_terrain_map=torch.tensor([0, 1, 1, 0], dtype=torch.long),
        terrain_loader=terrain_loader,
    )

    dones = torch.tensor([True, True, False, True])
    reached_goal = torch.tensor([True, False, True, False])

    metrics = wrapper._terrain_episode_metrics(dones, reached_goal)

    assert torch.allclose(metrics["terrain_0_d1_plane_success_rate"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(metrics["terrain_0_d1_plane_episode_count"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(metrics["terrain_1_d1_block_forest_success_rate"], torch.tensor([0.0]))
    assert torch.allclose(metrics["terrain_1_d1_block_forest_episode_count"], torch.tensor([1.0]))
