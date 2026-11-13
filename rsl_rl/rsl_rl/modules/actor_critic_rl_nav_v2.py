from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from utils.rl_nav_policy.model import RLNavConditionedQueryMapEncoder, build_mlp

from .actor_critic import get_activation


class ActorCriticRLNavV2(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        crop_size=8,
        token_dim=266,
        condition_dim=14,
        encoder_embed_dim=128,
        encoder_heads=4,
        encoder_hidden_dims=(256,),
        crop_encoder_hidden_dims=None,
        query_hidden_dims=None,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        activation="elu",
        init_noise_std=1.0,
        attention_dropout=0.0,
        **kwargs,
    ):
        if kwargs:
            print("ActorCriticRLNavV2.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))
        super().__init__()

        self.crop_size = int(crop_size)
        self.num_tokens = self.crop_size * self.crop_size
        self.token_dim = int(token_dim)
        self.condition_dim = int(condition_dim)
        expected_obs = self.num_tokens * self.token_dim + self.condition_dim
        if int(num_actor_obs) != expected_obs:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} does not match RL nav layout "
                f"{self.num_tokens} tokens * {self.token_dim} + {self.condition_dim} = {expected_obs}"
            )
        if int(num_critic_obs) != expected_obs:
            raise ValueError("ActorCriticRLNavV2 currently expects critic observations to match actor observations.")

        activation_module = get_activation(activation)
        encoder_kwargs = {
            "token_dim": self.token_dim,
            "condition_dim": self.condition_dim,
            "embed_dim": int(encoder_embed_dim),
            "num_heads": int(encoder_heads),
            "hidden_dims": tuple(int(v) for v in encoder_hidden_dims),
            "crop_hidden_dims": _optional_int_tuple(crop_encoder_hidden_dims),
            "query_hidden_dims": _optional_int_tuple(query_hidden_dims),
            "dropout": float(attention_dropout),
            "activation": activation_module,
        }
        self.actor_encoder = RLNavConditionedQueryMapEncoder(**encoder_kwargs)
        self.critic_encoder = RLNavConditionedQueryMapEncoder(**encoder_kwargs)
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

        print(f"RLNavV2 actor encoder: {self.actor_encoder}")
        print(f"RLNavV2 actor head: {self.actor}")
        print(f"RLNavV2 critic encoder: {self.critic_encoder}")
        print(f"RLNavV2 critic head: {self.critic}")

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

    def split_observation(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens_flat = observations[:, : self.num_tokens * self.token_dim]
        condition = observations[:, self.num_tokens * self.token_dim :]
        tokens = tokens_flat.view(observations.shape[0], self.num_tokens, self.token_dim)
        return tokens, condition

    def update_distribution(self, observations):
        tokens, condition = self.split_observation(observations)
        mean = self.actor(self.actor_encoder(tokens, condition))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        tokens, condition = self.split_observation(observations)
        return self.actor(self.actor_encoder(tokens, condition))

    def get_actor_attention_grid(self, observations, *, average_heads: bool = True):
        tokens, condition = self.split_observation(observations)
        _, weights = self.actor_encoder.forward_with_attention(
            tokens,
            condition,
            need_weights=True,
            average_attn_weights=bool(average_heads),
        )
        if weights is None:
            raise RuntimeError("Actor attention weights were not returned.")
        if average_heads:
            weights = weights.squeeze(1)
            return weights.reshape(observations.shape[0], self.crop_size, self.crop_size)
        weights = weights.squeeze(2)
        return weights.reshape(observations.shape[0], -1, self.crop_size, self.crop_size)

    def evaluate(self, critic_observations, **kwargs):
        tokens, condition = self.split_observation(critic_observations)
        return self.critic(self.critic_encoder(tokens, condition))


def _as_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def _optional_int_tuple(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    return _as_int_tuple(values)
