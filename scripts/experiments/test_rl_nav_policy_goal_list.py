from types import SimpleNamespace

import isaacgym  # noqa: F401
import torch
import pytest

from scripts.experiments.modules.vel_command_generator.rl_nav_policy import RLNavPolicyVelCommandGenerator


def make_generator(goal_xy_list, *, num_envs=1, start_index=0):
    generator = object.__new__(RLNavPolicyVelCommandGenerator)
    generator.env = SimpleNamespace(device=torch.device("cpu"), num_envs=num_envs)
    generator.cfg = SimpleNamespace(goal_source="list")
    generator.goal_xy = torch.zeros((num_envs, 2), dtype=torch.float32)
    generator.user_goal_xy = None
    generator.goal_xy_list = generator._validate_goal_xy_list(goal_xy_list)
    generator.goal_list_index_by_env = torch.full((num_envs,), start_index, dtype=torch.long)
    generator.synced = []
    generator._sync_env_task_goals = lambda env_ids, goal_xy: generator.synced.append(
        (env_ids.clone(), goal_xy.clone())
    )
    return generator


def test_next_goals_from_list_cycles_for_single_env():
    generator = make_generator([[1.0, 2.0], [3.0, 4.0]], start_index=0)
    env_ids = torch.tensor([0], dtype=torch.long)

    assert torch.equal(generator._next_goals_from_list(env_ids), torch.tensor([[1.0, 2.0]]))
    assert torch.equal(generator._next_goals_from_list(env_ids), torch.tensor([[3.0, 4.0]]))
    assert torch.equal(generator._next_goals_from_list(env_ids), torch.tensor([[1.0, 2.0]]))


def test_refresh_goals_with_list_source_advances_only_when_resampling():
    generator = make_generator([[1.0, 2.0], [3.0, 4.0]])
    env_ids = torch.tensor([0], dtype=torch.long)

    generator._refresh_goals(env_ids, resample=True)
    assert torch.equal(generator.goal_xy, torch.tensor([[1.0, 2.0]]))

    generator._refresh_goals(env_ids, resample=False)
    assert torch.equal(generator.goal_xy, torch.tensor([[1.0, 2.0]]))

    generator._refresh_goals(env_ids, resample=True)
    assert torch.equal(generator.goal_xy, torch.tensor([[3.0, 4.0]]))


def test_validate_goal_xy_list_rejects_bad_shape():
    generator = make_generator([[1.0, 2.0]])

    with pytest.raises(ValueError, match="goal_xy_list"):
        generator._validate_goal_xy_list([[1.0, 2.0, 3.0]])
