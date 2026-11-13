from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from utils.rl_nav_policy.model import build_mlp

from .actor_critic import get_activation


class ActorCriticRLNavConditionOnly(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        condition_dim=14,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRLNavConditionOnly.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.condition_dim = int(condition_dim)
        if int(num_actor_obs) != self.condition_dim:
            raise ValueError(
                f"num_actor_obs={num_actor_obs} does not match condition_dim={self.condition_dim} "
                "for condition-only RL nav."
            )
        if int(num_critic_obs) != self.condition_dim:
            raise ValueError(
                f"num_critic_obs={num_critic_obs} does not match condition_dim={self.condition_dim} "
                "for condition-only RL nav."
            )

        activation_module = get_activation(activation)
        self.actor = build_mlp(
            self.condition_dim,
            _as_int_tuple(actor_hidden_dims),
            int(num_actions),
            activation_module,
        )
        self.critic = build_mlp(
            self.condition_dim,
            _as_int_tuple(critic_hidden_dims),
            1,
            activation_module,
        )

        print(f"RLNav condition-only actor: {self.actor}")
        print(f"RLNav condition-only critic: {self.critic}")

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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


def _as_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)
