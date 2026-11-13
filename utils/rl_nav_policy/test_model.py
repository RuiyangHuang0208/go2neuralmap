import torch
from torch import nn

from rsl_rl.modules import (
    ActorCriticRLNav,
    ActorCriticRLNavConditionOnly,
    ActorCriticRLNavV2,
    ActorCriticRLNavV3,
    ActorCriticRLNavV3NoAttention,
)


def test_actor_critic_rl_nav_forward_shapes():
    crop_size = 2
    token_dim = 18
    condition_dim = 14
    num_obs = crop_size * crop_size * token_dim + condition_dim
    model = ActorCriticRLNav(
        num_obs,
        num_obs,
        3,
        crop_size=crop_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    actions = model.act(obs)
    values = model.evaluate(obs)

    assert actions.shape == (5, 3)
    assert values.shape == (5, 1)


def test_actor_critic_rl_nav_attention_grid_shape():
    crop_size = 2
    token_dim = 18
    condition_dim = 14
    num_obs = crop_size * crop_size * token_dim + condition_dim
    model = ActorCriticRLNav(
        num_obs,
        num_obs,
        3,
        crop_size=crop_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    attention_grid = model.get_actor_attention_grid(obs)

    assert attention_grid.shape == (5, crop_size, crop_size)
    assert torch.allclose(attention_grid.sum(dim=(1, 2)), torch.ones(5), atol=1.0e-5)


def test_actor_critic_rl_nav_v2_forward_shapes():
    crop_size = 2
    token_dim = 18
    condition_dim = 14
    num_obs = crop_size * crop_size * token_dim + condition_dim
    model = ActorCriticRLNavV2(
        num_obs,
        num_obs,
        3,
        crop_size=crop_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        crop_encoder_hidden_dims=[32],
        query_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    actions = model.act(obs)
    values = model.evaluate(obs)

    assert actions.shape == (5, 3)
    assert values.shape == (5, 1)


def test_actor_critic_rl_nav_v2_attention_grid_shape():
    crop_size = 2
    token_dim = 18
    condition_dim = 14
    num_obs = crop_size * crop_size * token_dim + condition_dim
    model = ActorCriticRLNavV2(
        num_obs,
        num_obs,
        3,
        crop_size=crop_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        crop_encoder_hidden_dims=[32],
        query_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    attention_grid = model.get_actor_attention_grid(obs)

    assert attention_grid.shape == (5, crop_size, crop_size)
    assert torch.allclose(attention_grid.sum(dim=(1, 2)), torch.ones(5), atol=1.0e-5)


def test_actor_critic_rl_nav_v3_forward_shapes():
    local_crop_size = 2
    global_pool_size = 3
    token_dim = 18
    condition_dim = 14
    num_obs = (local_crop_size * local_crop_size + global_pool_size * global_pool_size) * token_dim + condition_dim
    model = ActorCriticRLNavV3(
        num_obs,
        num_obs,
        3,
        local_crop_size=local_crop_size,
        global_pool_size=global_pool_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        local_crop_encoder_hidden_dims=[32],
        global_encoder_hidden_dims=[32],
        query_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    actions = model.act(obs)
    values = model.evaluate(obs)

    assert actions.shape == (5, 3)
    assert values.shape == (5, 1)


def test_actor_critic_rl_nav_v3_attention_grid_shape():
    local_crop_size = 2
    global_pool_size = 3
    token_dim = 18
    condition_dim = 14
    num_obs = (local_crop_size * local_crop_size + global_pool_size * global_pool_size) * token_dim + condition_dim
    model = ActorCriticRLNavV3(
        num_obs,
        num_obs,
        3,
        local_crop_size=local_crop_size,
        global_pool_size=global_pool_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        local_crop_encoder_hidden_dims=[32],
        global_encoder_hidden_dims=[32],
        query_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    attention_grid = model.get_actor_attention_grid(obs)

    assert attention_grid.shape == (5, local_crop_size, local_crop_size)
    assert torch.allclose(attention_grid.sum(dim=(1, 2)), torch.ones(5), atol=1.0e-5)


def test_actor_critic_rl_nav_v3_no_attention_forward_shapes():
    local_crop_size = 2
    global_pool_size = 3
    token_dim = 18
    condition_dim = 14
    num_obs = (local_crop_size * local_crop_size + global_pool_size * global_pool_size) * token_dim + condition_dim
    model = ActorCriticRLNavV3NoAttention(
        num_obs,
        num_obs,
        3,
        local_crop_size=local_crop_size,
        global_pool_size=global_pool_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_heads=4,
        encoder_hidden_dims=[32],
        local_context_hidden_dims=[32],
        local_crop_encoder_hidden_dims=[32],
        global_encoder_hidden_dims=[32],
        query_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    actions = model.act(obs)
    values = model.evaluate(obs)

    assert actions.shape == (5, 3)
    assert values.shape == (5, 1)


def test_actor_critic_rl_nav_v3_no_attention_grid_is_uniform():
    local_crop_size = 2
    global_pool_size = 3
    token_dim = 18
    condition_dim = 14
    num_obs = (local_crop_size * local_crop_size + global_pool_size * global_pool_size) * token_dim + condition_dim
    model = ActorCriticRLNavV3NoAttention(
        num_obs,
        num_obs,
        3,
        local_crop_size=local_crop_size,
        global_pool_size=global_pool_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_hidden_dims=[32],
        local_context_hidden_dims=[32],
        local_crop_encoder_hidden_dims=[32],
        global_encoder_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, num_obs), dtype=torch.float32)

    grid = model.get_actor_attention_grid(obs)
    per_head_grid = model.get_actor_attention_grid(obs, average_heads=False)

    assert grid.shape == (5, local_crop_size, local_crop_size)
    assert per_head_grid.shape == (5, 1, local_crop_size, local_crop_size)
    assert torch.allclose(grid.sum(dim=(1, 2)), torch.ones(5), atol=1.0e-5)
    assert torch.allclose(grid, torch.full_like(grid, 1.0 / (local_crop_size * local_crop_size)))


def test_actor_critic_rl_nav_v3_no_attention_has_no_mha():
    local_crop_size = 2
    global_pool_size = 3
    token_dim = 18
    condition_dim = 14
    num_obs = (local_crop_size * local_crop_size + global_pool_size * global_pool_size) * token_dim + condition_dim
    model = ActorCriticRLNavV3NoAttention(
        num_obs,
        num_obs,
        3,
        local_crop_size=local_crop_size,
        global_pool_size=global_pool_size,
        token_dim=token_dim,
        condition_dim=condition_dim,
        encoder_embed_dim=16,
        encoder_hidden_dims=[32],
        local_crop_encoder_hidden_dims=[32],
        global_encoder_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )

    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())


def test_actor_critic_rl_nav_condition_only_forward_shapes():
    condition_dim = 14
    model = ActorCriticRLNavConditionOnly(
        condition_dim,
        condition_dim,
        3,
        condition_dim=condition_dim,
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
    )
    obs = torch.zeros((5, condition_dim), dtype=torch.float32)

    actions = model.act(obs)
    values = model.evaluate(obs)

    assert actions.shape == (5, 3)
    assert values.shape == (5, 1)
