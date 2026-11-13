from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from utils.rl_nav_policy.model import RLNavDualScalePooledMapEncoder, build_mlp

from .actor_critic import get_activation


class ActorCriticRLNavV3NoAttention(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        local_crop_size=8,
        global_pool_size=12,
        crop_size=None,
        token_dim=266,
        condition_dim=14,
        encoder_embed_dim=128,
        encoder_heads=4,
        encoder_hidden_dims=(256,),
        local_context_hidden_dims=None,
        local_crop_encoder_hidden_dims=None,
        global_encoder_hidden_dims=None,
        query_hidden_dims=None,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        activation="elu",
        init_noise_std=1.0,
        attention_dropout=0.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRLNavV3NoAttention.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        del encoder_heads, query_hidden_dims, attention_dropout

        if crop_size is not None and int(local_crop_size) == 8:
            local_crop_size = crop_size
        self.local_crop_size = int(local_crop_size)
        self.global_pool_size = int(global_pool_size)
        self.local_num_tokens = self.local_crop_size * self.local_crop_size
        self.global_num_tokens = self.global_pool_size * self.global_pool_size
        self.token_dim = int(token_dim)
        self.condition_dim = int(condition_dim)
        expected_obs = (self.local_num_tokens + self.global_num_tokens) * self.token_dim + self.condition_dim
        if int(num_actor_obs) != expected_obs:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} does not match RL nav V3 no-attention layout "
                f"({self.local_num_tokens} local + {self.global_num_tokens} global) tokens "
                f"* {self.token_dim} + {self.condition_dim} = {expected_obs}"
            )
        if int(num_critic_obs) != expected_obs:
            raise ValueError(
                "ActorCriticRLNavV3NoAttention currently expects critic observations to match actor observations."
            )

        activation_module = get_activation(activation)
        encoder_kwargs = {
            "token_dim": self.token_dim,
            "condition_dim": self.condition_dim,
            "embed_dim": int(encoder_embed_dim),
            "hidden_dims": tuple(int(v) for v in encoder_hidden_dims),
            "local_context_hidden_dims": _optional_int_tuple(local_context_hidden_dims),
            "local_crop_hidden_dims": _optional_int_tuple(local_crop_encoder_hidden_dims),
            "global_hidden_dims": _optional_int_tuple(global_encoder_hidden_dims),
            "activation": activation_module,
        }
        self.actor_encoder = RLNavDualScalePooledMapEncoder(**encoder_kwargs)
        self.critic_encoder = RLNavDualScalePooledMapEncoder(**encoder_kwargs)
        self.actor = build_mlp(
            self.actor_encoder.output_dim,
            _as_int_tuple(actor_hidden_dims),
            int(num_actions),
            activation_module,
        )
        self.critic = build_mlp(
            self.critic_encoder.output_dim,
            _as_int_tuple(critic_hidden_dims),
            1,
            activation_module,
        )

        print("RLNavV3NoAttention uses pooled local context and does not use MHA/query attention.")
        print(f"RLNavV3NoAttention actor encoder: {self.actor_encoder}")
        print(f"RLNavV3NoAttention actor head: {self.actor}")
        print(f"RLNavV3NoAttention critic encoder: {self.critic_encoder}")
        print(f"RLNavV3NoAttention critic head: {self.critic}")

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def split_observation(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_width = self.local_num_tokens * self.token_dim
        global_width = self.global_num_tokens * self.token_dim
        local_flat = observations[:, :local_width]
        global_flat = observations[:, local_width : local_width + global_width]
        condition = observations[:, local_width + global_width :]
        local_tokens = local_flat.view(observations.shape[0], self.local_num_tokens, self.token_dim)
        global_tokens = global_flat.view(observations.shape[0], self.global_num_tokens, self.token_dim)
        return local_tokens, global_tokens, condition

    def update_distribution(self, observations):
        local_tokens, global_tokens, condition = self.split_observation(observations)
        mean = self.actor(self.actor_encoder(local_tokens, global_tokens, condition))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        local_tokens, global_tokens, condition = self.split_observation(observations)
        return self.actor(self.actor_encoder(local_tokens, global_tokens, condition))

    def get_actor_attention_grid(self, observations, *, average_heads: bool = True):
        value = 1.0 / float(self.local_num_tokens)
        if average_heads:
            return observations.new_full(
                (observations.shape[0], self.local_crop_size, self.local_crop_size),
                value,
            )
        return observations.new_full(
            (observations.shape[0], 1, self.local_crop_size, self.local_crop_size),
            value,
        )

    def evaluate(self, critic_observations, **kwargs):
        local_tokens, global_tokens, condition = self.split_observation(critic_observations)
        return self.critic(self.critic_encoder(local_tokens, global_tokens, condition))


def _as_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _optional_int_tuple(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return _as_int_tuple(values)
