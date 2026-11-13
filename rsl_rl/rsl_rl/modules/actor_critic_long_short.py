# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
# from rsl_rl.modules.dual_hist_encoder import DualHistEncoder
# from rsl_rl.modules.transformer_modelling import DecisionTransformer
from rsl_rl.modules.former.transformer_in_use import RCTransformer  # , RGTransformer
# from rsl_rl.modules.cnn_history_encoder import CNNHistoryEncoder
from enum import Enum

class PolicyMode(str, Enum):
    CURRENT = "current"
    CUR_SHORT = "current+short"
    CUR_LONG = "current+long"
    CUR_SHORT_LONG = "current+short+long"
    SHORT_LONG = "short+long"
    RECENT4 = "recent4"
    RECENT4_LONG = "recent4+long"

class GateMix(nn.Module):
    """
    Soft-MoE fuse: produces alpha over {long, short} either per-dimension or scalar.
    Options: temperature, epsilon (avoid saturated 0/1), pre/post LayerNorm.
    """
    def __init__(self, L, hidden=None, temp=0.5, epsilon=0.05,
                 per_dim=True, pre_ln=True, post_ln=True, activation=nn.ELU()):
        super().__init__()
        self.L = L
        self.temp = temp
        self.epsilon = epsilon
        self.per_dim = per_dim
        self.ln_long = nn.LayerNorm(L) if pre_ln else nn.Identity()
        self.ln_short = nn.LayerNorm(L) if pre_ln else nn.Identity()
        self.ln_out = nn.LayerNorm(L) if post_ln else nn.Identity()
        H = hidden or (2 * L)

        self.mlp = nn.Sequential(
            nn.Linear(2 * L, H),
            activation,
            nn.Linear(H, 2 * L if per_dim else 2)
        )

    def forward(self, l, s):
        # normalize each stream before mixing
        l_n = self.ln_long(l)
        s_n = self.ln_short(s)
        h = torch.cat([s_n, l_n], dim=-1)

        logits = self.mlp(h)                           # [B,2*L] or [B,2]
        if self.per_dim:
            logits = logits.view(l.shape[0], 2, self.L)  # [B,2,L]
            alpha = torch.softmax(logits / self.temp, dim=1)  # [B,2,L]
            # avoid dead paths
            alpha = (1 - 2*self.epsilon) * alpha + self.epsilon
            z = alpha[:, 0, :] * l_n + alpha[:, 1, :] * s_n   # long vs short order matches concat above
        else:
            alpha = torch.softmax(logits / self.temp, dim=-1)  # [B,2]
            alpha = (1 - 2*self.epsilon) * alpha + self.epsilon
            z = alpha[:, :1] * l_n + alpha[:, 1:] * s_n        # broadcast over dim

        z = self.ln_out(z)
        return z, alpha



class TCNEncoder(nn.Module):
    """
    An implementation of the TCN architecture described in the image and text.
    It processes a history of observations using a series of 1D convolutional
    layers with specific dilations and strides.
    """
    def __init__(self, input_dim, channels=[34, 34, 34], kernel_size=5, output_dim=64):
        """
        Args:
            input_dim (int): The feature dimension of a single observation in the history.
            channels (list): A list of output channels for each of the three main conv layers.
            kernel_size (int): The size of the convolutional kernel.
            output_dim (int): The final output dimension after the MLP head.
        """
        super().__init__()

        # Layer parameters from the description
        dilations = [1, 2, 4] # Dilation base of 2 -> 2^0, 2^1, 2^2
        stride = 2

        layers = []
        current_channels = input_dim

        # Create the three pairs of (dilation conv, stride conv)
        for i, (dilation, out_channels) in enumerate(zip(dilations, channels)):
            # Causal padding for the dilated convolution
            pad_dilation = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(current_channels, out_channels, kernel_size, padding=pad_dilation))
            layers.append(nn.ReLU())

            # Causal padding for the strided convolution
            pad_stride = (kernel_size - 1) 
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=pad_stride))
            layers.append(nn.ReLU())
            
            current_channels = out_channels

        self.convolutional_net = nn.Sequential(*layers)

        # The final part of the network: a linear layer and a tanh activation
        # We need a dummy forward pass to figure out the flattened size
        with torch.no_grad():
            # Assuming a history length of 100 for this calculation
            dummy_input = torch.zeros(1, input_dim, 100)
            dummy_output = self.convolutional_net(dummy_input)
            flattened_size = dummy_output.flatten(1).shape[1]

        self.mlp_head = nn.Sequential(
            nn.Linear(flattened_size, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Performs the forward pass.
        
        Args:
            x (torch.Tensor): The input history tensor.
                              Expected shape: (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: The encoded representation of the history.
                          Shape: (batch_size, output_dim)
        """
        # Conv1d expects (batch_size, channels, sequence_length)
        # So we permute the input dimensions
        x = x.permute(0, 2, 1)

        # Pass through the convolutional layers
        conv_out = self.convolutional_net(x)

        # To remove the padding added for causality
        conv_out = conv_out[:, :, :-1]

        # Flatten the output for the linear layer
        flattened = conv_out.flatten(start_dim=1)

        # Pass through the final MLP head
        output = self.mlp_head(flattened)

        return output


class CNNHistoryEncoderV2(nn.Module):
    """
    Input:  x ∈ [B, T=100, P=42]  (continuous 100 steps @ 50 Hz)
    Output: z ∈ [B, L=32]
    Design:
      - Conv1d k=5, stride=2, padding=2 (learned low-pass + decimation): 100 → 50
      - Small conv stack: 50→23→10→8 (all channels = 32)
      - AdaptiveAvgPool1d(1) → Linear(32 → 32)  (length-invariant head, no tanh)
      - GroupNorm(1,C) after each conv for RL stability
    """
    def __init__(self, output_size=32, activation=nn.ELU(), dropout_p=0.0):
        super().__init__()
        P = 42
        C1 = 32
        C2 = 16
        C3 = 8
        C4 = 4
        self.act = activation

        def GN(C): return nn.GroupNorm(1, C)

        # 1) learned decimation: 100 -> 50
        # self.conv1 = nn.Conv1d(P, C1, kernel_size=6, stride=2, padding=0) # 100->48
        # # 2) reductions: 50->23->10->8
        # self.conv2 = nn.Conv1d(C1, C2, kernel_size=4, stride=2, padding=0)  # 32->15
        # # self.norm2 = GN(C2)

        # self.conv3 = nn.Conv1d(C2, C3, kernel_size=3, stride=2, padding=0)  # 15->6


        # self.conv1 = nn.Conv1d(P, C1, kernel_size=6, stride=3, padding=0) # 100->32
        # # self.norm1 = GN(C1)

        # # 2) reductions: 50->23->10->8
        # self.conv2 = nn.Conv1d(C1, C2, kernel_size=5, stride=3, padding=0)  # 32->10
        # # self.norm2 = GN(C2)

        # self.conv3 = nn.Conv1d(C2, C3, kernel_size=4, stride=2, padding=0)  # 10->4

        # self.conv4 = nn.Conv1d(C3, C4, kernel_size=3, stride=1, padding=0)  # 6->4
        # self.norm4 = GN(C4)

        self.conv1 = nn.Conv1d(P, C1, kernel_size=6, stride=3, padding=0) # 100->32
        # self.norm1 = GN(C1)

        # 2) reductions: 50->23->10->8
        self.conv2 = nn.Conv1d(C1, C2, kernel_size=4, stride=2, padding=0)  # 32->15
        # self.norm2 = GN(C2)

        # self.conv3 = nn.Conv1d(C2, C3, kernel_size=4, stride=2, padding=0)  # 10->4

        self.flatten = nn.Flatten()

        self.drop = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # 3) length-invariant head: [B, 32, ~8] -> [B, 32]
        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(1),
        #     nn.Linear(C4, output_size)  # 32 -> 32
        # )
        # self.head = nn.Linear(240, output_size)  # 32 -> 32

        self.head = nn.Sequential(
            nn.Linear(240, 32),
            # nn.ELU(),
            # nn.Linear(64, output_size)
            # nn.Linear(C4, output_size)  # 32 -> 32
        )
        

    def forward(self, x):
        # x: [B, T, P]  e.g., [B,100,42]
        x = x.permute(0, 2, 1)                    # -> [B, P, T]
        # y = self.act(self.norm1(self.conv1(x)));  y = self.drop(y)    # -> [B, 32, 50]
        # y = self.act(self.norm2(self.conv2(y)));  y = self.drop(y)    # -> [B, 32, 23]
        # y = self.act(self.norm3(self.conv3(y)));  y = self.drop(y)    # -> [B, 32, 10]
        # y = self.act(self.norm4(self.conv4(y)))                       # -> [B, 32,  8]
        # z = self.head(y)                                               # -> [B, 32]
        y = self.act(self.conv1(x));     # -> [B, 32, 50]
        y = self.act(self.conv2(y));     # -> [B, 16, 23]
        # y = self.act(self.conv3(y));     # -> [B, 8, 10]
        # y = self.act(self.conv4(y));     # -> [B, 8,  8]
        y = self.flatten(y)              # -> [B, 8*8=64]
        z = self.head(y) 
        return z
    
class Actor(nn.Module):
    def __init__(self, 
                 num_proprio_obs,
                 num_actions,
                 num_long_history_length,
                 num_short_history_length,
                 dim_long_history_latent,
                 actor_hidden_dims,
                 activation,
                 policy_mode="current+short",
                 long_encoder_type="cnn"):
        super().__init__()

        self.policy_mode = PolicyMode(policy_mode)
        self.long_encoder_type = long_encoder_type  # "cnn" | "transformer"
        
        self.long_history_length  = num_long_history_length   # recommend 100 for continuous window
        self.short_history_length = num_short_history_length
        self.dim_long_history_latent = dim_long_history_latent  # == L == 32
        self.proprio_obs_dim = num_proprio_obs                 # == P == 42
        self.num_actions = num_actions

        # --- flags for which ingredients to feed into the actor backbone ---
        self.use_current = self.policy_mode in {
            PolicyMode.CURRENT, PolicyMode.CUR_SHORT, PolicyMode.CUR_LONG, PolicyMode.CUR_SHORT_LONG
        }
        self.use_short = self.policy_mode in {
            PolicyMode.CUR_SHORT, PolicyMode.CUR_SHORT_LONG, PolicyMode.SHORT_LONG
        }
        self.use_long = self.policy_mode in {
            PolicyMode.CUR_LONG, PolicyMode.CUR_SHORT_LONG, PolicyMode.RECENT4_LONG, PolicyMode.SHORT_LONG
        }
        self.use_recent4 = self.policy_mode in {
            PolicyMode.RECENT4, PolicyMode.RECENT4_LONG
        }

        if self.use_long:
            if self.long_encoder_type == "cnn":
                self.long_encoder = CNNHistoryEncoderV2(output_size=self.dim_long_history_latent,
                                                        activation=activation, dropout_p=0.0)
            elif self.long_encoder_type == "transformer":
                # Example config; adjust as needed
                seq_length = self.long_history_length
                obs_act_dim = num_proprio_obs
                # act_dim = 12
                # obs_act_dim = obs_dim + act_dim
                emb_dim = 32
                heads = 2
                depth = 1
                output_dim = dim_long_history_latent
                causal = False
                self.long_encoder = RCTransformer(
                    emb=32, heads=2, depth=1,
                    seq_length=self.long_history_length,
                    obs_act_dim=self.proprio_obs_dim,
                    output_dim=self.dim_long_history_latent,
                    causal=False, dropout=0.0,
                )
            else:
                raise ValueError(f"Unknown long_encoder_type: {self.long_encoder_type}")
        else:
            self.long_encoder = None

        # --- short projection MLP (only if needed) ---
        # NOTE: expects last 6 frames → 6*P features. Ensure short_history_length >= 6
        if self.use_short:
            H_1, H_2, L = 128, 64, self.dim_long_history_latent
            self.short_proj = nn.Sequential(
                nn.Linear(6 * self.proprio_obs_dim, H_1),
                activation,
                nn.Linear(H_1, H_2),
                activation,
                nn.Linear(H_2, L)
            )
        else:
            self.short_proj = None

        # --- compute actor backbone input dim from selected ingredients ---
        # Always include command(3)
        input_dim = 3
        if self.use_current:
            input_dim += (self.proprio_obs_dim + 3)  # current_obs is (P+3) in your layout
        if self.use_short:
            input_dim += self.dim_long_history_latent  # short_proj (L)
        if self.use_long:
            input_dim += self.dim_long_history_latent  # long latent (L)
        if self.use_recent4:
            input_dim += (4 * self.proprio_obs_dim)    # concatenated 4 frames (4P)

        layers = [nn.Linear(input_dim, actor_hidden_dims[0]), activation]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                layers.append(nn.Linear(actor_hidden_dims[l], self.num_actions))
            else:
                layers += [nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l+1]), activation]
        self.actor_backbone = nn.Sequential(*layers)

    def _compute_latents(self, obs):
        """
        Slices always assume your original full observation layout so you don't
        need to change the env. We just compute what we need, based on flags.

        Returns dict with only the pieces the forward() may use.
        """
        B = obs.shape[0]; P = self.proprio_obs_dim
        terrain_one_hot_dim = 1

        expected_obs_dim = 3 + P + terrain_one_hot_dim + 3 + \
                           self.short_history_length * P + \
                           self.long_history_length  * P
        assert obs.shape[1] == expected_obs_dim, \
            f"Expected obs.shape[1]={expected_obs_dim}, got {obs.shape[1]}"

        command_obs = obs[:, :3]
        current_obs = obs[:, 3 : 3+P+3]  # (B, P+3)

        flat_short = obs[:, 3+P+terrain_one_hot_dim+3 :
                            3+P+terrain_one_hot_dim+3 + self.short_history_length*P]
        short_seq  = flat_short.view(B, self.short_history_length, P)  # includes current step

        flat_long  = obs[:, 3+P+terrain_one_hot_dim+3 + self.short_history_length*P :]
        long_seq   = flat_long.view(B, self.long_history_length,  P)

        out = {"command_obs": command_obs}

        if self.use_current:
            out["current_obs"] = current_obs

        if self.use_recent4 or self.use_short:
            # recent_4 always available if needed
            recent_4   = short_seq[:, -4:, :].contiguous().view(B, -1)  # (B, 4P)
            out["recent_4"] = recent_4

        if self.use_short:
            # assumes short_history_length >= 6
            recent_6   = short_seq[:, -6:, :].contiguous().view(B, -1)  # (B, 6P)
            out["short_proj"] = self.short_proj(recent_6)

        if self.use_long:
            if self.long_encoder is None:
                raise RuntimeError("policy_mode requires long encoder, but it's not initialized.")
            out["latent_long"] = self.long_encoder(long_seq)

        return out


    def forward(self, obs, return_latent=False):
        lat = self._compute_latents(obs)

        x_parts = [lat["command_obs"]]

        if self.policy_mode == PolicyMode.CURRENT:
            x_parts += [lat["current_obs"]]

        elif self.policy_mode == PolicyMode.CUR_SHORT:
            x_parts += [lat["current_obs"], lat["short_proj"]]

        elif self.policy_mode == PolicyMode.CUR_LONG:
            x_parts += [lat["current_obs"], lat["latent_long"]]

        elif self.policy_mode == PolicyMode.CUR_SHORT_LONG:
            x_parts += [lat["current_obs"], lat["short_proj"], lat["latent_long"]]

        elif self.policy_mode == PolicyMode.RECENT4:
            x_parts += [lat["recent_4"]]

        elif self.policy_mode == PolicyMode.RECENT4_LONG:
            x_parts += [lat["recent_4"], lat["latent_long"]]
        elif self.policy_mode == PolicyMode.SHORT_LONG:
            x_parts += [lat["short_proj"], lat["latent_long"]]

        else:
            raise ValueError(f"Unhandled policy_mode: {self.policy_mode}")

        x = torch.cat(x_parts, dim=-1)
        actions = self.actor_backbone(x)
        if return_latent:
            # Return whatever is present, falling back to None for absent parts
            return actions, lat.get("short_proj", None), lat.get("latent_long", None)
        return actions
    
    # def forward(self, obs):
    #     lat = self._compute_latents(obs)

    #     x_parts = [lat["command_obs"]]

    #     # --- This is the new, JIT-friendly logic ---
    #     # Use the boolean flags set in __init__ instead of Enum comparisons
    #     if self.use_current:
    #         x_parts.append(lat["current_obs"])
        
    #     if self.use_short:
    #         x_parts.append(lat["short_proj"])

    #     if self.use_long:
    #         x_parts.append(lat["latent_long"])

    #     if self.use_recent4:
    #         x_parts.append(lat["recent_4"])
    #     # --- End of new logic ---

    #     x = torch.cat(x_parts, dim=-1)
    #     actions = self.actor_backbone(x)
        
    #     # if return_latent:
    #     #     # Return whatever is present, falling back to None for absent parts
    #     #     return actions, lat.get("short_proj", None), lat.get("latent_long", None)
    #     return actions


class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_proprio_obs,
                        num_critic_obs,
                        num_actions,
                        num_long_history_length,
                        num_short_history_length,
                        dim_long_history_latent,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        privileged_encoder_hidden_dims=[256,128],
                        activation='elu',
                        init_noise_std=1.0,
                        policy_mode="current+short",
                        long_encoder_type="cnn",
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.long_history_length  = num_long_history_length
        self.short_history_length = num_short_history_length
        self.dim_long_history_latent = dim_long_history_latent
        self.proprio_obs_dim = num_proprio_obs
        self.num_actions = num_actions

        # critic MLP input = (critic_obs without long history) + long latent (L)
        mlp_input_dim_c = num_critic_obs #+ self.dim_long_history_latent #- self.long_history_length * self.proprio_obs_dim + self.dim_long_history_latent
        print(f"mlp_input_dim_c: {mlp_input_dim_c}")
        print(f"Policy mode: {policy_mode} | Long encoder: {long_encoder_type}")
        print(f"Critic MLP input dim: {mlp_input_dim_c}")

        # ---- policy ----
        self.actor = Actor(
            num_proprio_obs,
            num_actions,
            num_long_history_length,
            num_short_history_length,
            dim_long_history_latent,
            actor_hidden_dims,
            activation,
            policy_mode=policy_mode,
            long_encoder_type=long_encoder_type
        )



        # ---- value function ----
        critic_layers = [nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]), activation]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]), activation]
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

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
        mean = self.actor(observations)                # Actor.forward returns actions only
        mean = torch.clamp(mean, -100, 100)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            print(f"[NaN/Inf Mean] in update_distribution, obs stats:")
            print(f"mean: {observations.mean().item():.4f}, std: {observations.std().item():.4f}")
            raise ValueError("Actor mean has NaNs or Infs!")
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        actions = self.distribution.sample()
        return torch.clamp(actions, -100, 100)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        """
        Critic input = [ critic_observations_without_long_history , long_latent_from_CNN ]
        The long latent is computed with no grad, using the same CNN as the actor.
        """
        # B = critic_observations.shape[0]
        # P = self.proprio_obs_dim
        # history_size = self.long_history_length * P

        # # last history_size dims are the flattened long history [B, T_l * P]
        # flat_long = critic_observations[:, -history_size:].contiguous()
        # long_seq  = flat_long.view(B, self.long_history_length, P)

        # with torch.no_grad():   # do not train long encoder via critic loss
        #     latent_long = self.actor.long_encoder(long_seq)   # (B, L)

        # critic_input = torch.cat([critic_observations[:, :-history_size], latent_long], dim=-1)

        # critic_input = critic_observations[:, :-history_size]
        critic_input = critic_observations
        value = self.critic(critic_input)
        return value

    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
