import math

import torch

from utils.residual_policy.command_mixer import CommandMixerConfig, mix_residual_commands, wrap_pi


def test_mix_residual_commands_shape_and_scale():
    cfg = CommandMixerConfig(residual_horizon=2, residual_vxy_scale=0.3, residual_heading_scale=0.4)
    base = torch.zeros(3, 2, 3)
    actions = torch.ones(3, 6)

    mixed = mix_residual_commands(base, actions, cfg)

    assert mixed.shape == (3, 2, 3)
    assert torch.allclose(mixed[..., 0], torch.full((3, 2), 0.3))
    assert torch.allclose(mixed[..., 1], torch.full((3, 2), 0.3))
    assert torch.allclose(mixed[..., 2], torch.full((3, 2), 0.4))


def test_mix_residual_commands_clamps_velocity_and_wraps_heading():
    cfg = CommandMixerConfig(
        residual_horizon=1,
        residual_vxy_scale=1.0,
        residual_heading_scale=1.0,
        max_forward_speed=1.5,
        max_lateral_speed=0.5,
    )
    base = torch.tensor([[[1.0, -0.25, math.pi - 0.1]]])
    actions = torch.tensor([[10.0, -10.0, 0.3]])

    mixed = mix_residual_commands(base, actions, cfg)

    assert torch.allclose(mixed[..., 0], torch.tensor([[1.5]]))
    assert torch.allclose(mixed[..., 1], torch.tensor([[-0.5]]))
    assert -math.pi <= float(mixed[0, 0, 2]) <= math.pi
    assert torch.allclose(mixed[0, 0, 2], wrap_pi(torch.tensor(math.pi + 0.2)))


def test_mix_residual_commands_clips_residual_before_final_velocity_limits():
    cfg = CommandMixerConfig(
        residual_horizon=1,
        residual_vxy_scale=1.0,
        residual_heading_scale=1.0,
        residual_vel_clip=0.2,
        residual_heading_clip=0.3,
    )
    base = torch.tensor([[[0.8, -0.1, 0.0]]])
    actions = torch.tensor([[10.0, -10.0, 10.0]])

    mixed, info = mix_residual_commands(base, actions, cfg, return_info=True)

    assert torch.allclose(mixed[..., 0], torch.tensor([[1.0]]))
    assert torch.allclose(mixed[..., 1], torch.tensor([[-0.3]]))
    assert torch.allclose(mixed[..., 2], torch.tensor([[0.3]]))
    assert torch.allclose(info["residual_delta_v"], torch.tensor([[[0.2, -0.2]]]))
    assert torch.allclose(info["heading_delta"], torch.tensor([[0.3]]))


def test_mix_residual_commands_scale_velocity_mode():
    cfg = CommandMixerConfig(
        residual_horizon=1,
        mix_mode="scale_velocity",
        residual_vxy_scale=0.5,
        min_velocity_scale=0.3,
        max_velocity_scale=1.2,
    )
    base = torch.tensor([[[1.0, 0.4, 0.1]], [[1.0, 0.4, 0.1]]])
    actions = torch.tensor([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    mixed, info = mix_residual_commands(base, actions, cfg, return_info=True)

    assert torch.allclose(mixed[:, 0, 0], torch.tensor([0.3, 1.2]))
    assert torch.allclose(mixed[:, 0, 1], torch.tensor([0.12, 0.48]))
    assert torch.allclose(info["velocity_scale"], torch.tensor([[0.3], [1.2]]))
