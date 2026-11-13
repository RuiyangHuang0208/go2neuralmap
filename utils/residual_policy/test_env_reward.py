from types import SimpleNamespace

import torch

from utils.residual_policy.env import ResidualEnvConfig, ResidualPolicyEnv, ResidualRewardConfig


def test_compute_reward_uses_dense_progress_and_split_residual_penalties():
    env = ResidualPolicyEnv.__new__(ResidualPolicyEnv)
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.cfg = ResidualEnvConfig(
        reward=ResidualRewardConfig(
            variant="shaped",
            progress_scale=2.0,
            progress_positive_only=True,
            reached_goal_reward=10.0,
            collision_penalty=1.0,
            termination_penalty=10.0,
            residual_vel_penalty_scale=0.5,
            residual_heading_penalty_scale=0.25,
            anti_cancellation_penalty_scale=0.2,
            time_penalty=0.01,
        )
    )
    actions = torch.zeros(2, 6)
    mixer_info = {
        "residual_delta_v": torch.tensor(
            [
                [[0.2, 0.0], [0.0, 0.2]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
        "heading_delta": torch.tensor([[0.1, -0.1], [0.0, 0.0]]),
        "anti_cancellation": torch.tensor([[0.2, 0.0], [0.0, 0.0]]),
    }

    reward = env._compute_reward(
        actions=actions,
        previous_distance=torch.tensor([2.0, 2.0]),
        current_distance=torch.tensor([1.5, 2.5]),
        reached_goal=torch.tensor([False, False]),
        collision=torch.tensor([False, True]),
        terminated=torch.tensor([False, False]),
        mixer_info=mixer_info,
    )

    expected_first = 1.0 - 0.01 - 0.5 * 0.04 - 0.25 * 0.01 - 0.2 * 0.1
    expected_second = 0.0 - 1.0 - 0.01
    assert torch.allclose(reward, torch.tensor([expected_first, expected_second]))


def test_compute_reward_keeps_legacy_raw_action_penalty_when_new_terms_disabled():
    env = ResidualPolicyEnv.__new__(ResidualPolicyEnv)
    env.num_envs = 1
    env.device = torch.device("cpu")
    env.cfg = SimpleNamespace(reward=ResidualRewardConfig(residual_penalty_scale=0.1))

    reward = env._compute_reward(
        actions=torch.ones(1, 6),
        previous_distance=torch.tensor([2.0]),
        current_distance=torch.tensor([1.0]),
        reached_goal=torch.tensor([False]),
        collision=torch.tensor([False]),
        terminated=torch.tensor([False]),
        mixer_info={},
    )

    assert torch.allclose(reward, torch.tensor([-0.1]))
