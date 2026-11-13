import torch

from utils.rl_nav_policy.command import RLNavCommandConfig, scale_rl_nav_actions


def test_scale_rl_nav_actions_uses_asymmetric_forward_backward_limits():
    cfg = RLNavCommandConfig(
        max_forward_speed=1.5,
        max_backward_speed=0.5,
        max_lateral_speed=0.4,
        max_yaw_rate=0.8,
    )

    commands = scale_rl_nav_actions(torch.tensor([[1.0, -1.0, 0.5], [-1.0, 0.5, -1.0]]), cfg)

    assert torch.allclose(commands[0], torch.tensor([1.5, -0.4, 0.4]))
    assert torch.allclose(commands[1], torch.tensor([-0.5, 0.2, -0.8]))


def test_scale_rl_nav_actions_clips_normalized_actions():
    cfg = RLNavCommandConfig(max_forward_speed=1.0, max_backward_speed=1.0, max_lateral_speed=1.0, max_yaw_rate=1.0)

    commands = scale_rl_nav_actions(torch.tensor([[2.0, -3.0, 4.0]]), cfg)

    assert torch.allclose(commands, torch.tensor([[1.0, -1.0, 1.0]]))
