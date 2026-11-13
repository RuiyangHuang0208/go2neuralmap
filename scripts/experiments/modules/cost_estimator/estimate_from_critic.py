from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from .cost_estimator import CostEstimator, CostEstimatorConfig
from loguru import logger


@dataclass
class EstimateFromCriticConfig(CostEstimatorConfig):
    ckpt_path: str = "/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/notebooks/critic_weights.pt"
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])

    add_env_cost: bool = True


class Critic(nn.Module):
    def __init__(self, critic_hidden_dims):
        super().__init__()
        activation = nn.ELU()
        critic_layers = [nn.Linear(277, critic_hidden_dims[0]), activation]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]), activation]
        self.critic = nn.Sequential(*critic_layers)

    def load_weights(self,
                     ckpt_path="/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/notebooks/critic_weights.pt",
                     map_location=None):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(state_dict)

    def forward(self, obs):
        return self.critic(obs)


class EstimateFromCritic(CostEstimator):
    def __init__(self, cfg: EstimateFromCriticConfig, env_handle, **kwargs):
        self.cfg = cfg

        self.logger = logger.bind(module_name="CostEstimator.from_critic")
        self.env_handle = env_handle

        self.critic = Critic(self.cfg.critic_hidden_dims)
        self.critic.load_weights(self.cfg.ckpt_path, map_location=env_handle.device)

        self.critic.to(env_handle.device)

    def compute_cost(self, **kwargs):
        """
        return: cost, shape=(num_envs,)
        """
        priv_obs = kwargs.get("privileged_obs")
        if priv_obs is None:
            self.logger.error("Priv Obs not passed")
            raise ValueError("Priv Obs not passed")
        cost = self.critic.forward(priv_obs).squeeze(-1)
        cost = 100. / cost
        if self.cfg.add_env_cost:
            cost = cost + self.env_handle.step_cost.view_as(cost)
        return cost
