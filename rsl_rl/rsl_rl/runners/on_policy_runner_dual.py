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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch

import wandb

from rsl_rl.algorithms import PPO
# from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.modules.actor_critic_long_short import ActorCritic
from rsl_rl.env import VecEnv
from rsl_rl.utils.wandb_utils import WandbSummaryWriter

import math

def safe_add_scalar(writer, tag, value, step):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if value is not None and not math.isnan(value) and not math.isinf(value):
        writer.add_scalar(tag, value, step)
    else:
        print(f"[Warning] Skipped logging {tag} at step {step}: value is {value}")

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.train_velestimator = False
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        # The eval function lets a Python program run Python code within itself.

        num_long_history_length = self.env.long_history_length
        num_short_history_length = self.env.short_history_length
        dim_long_history_latent = self.env.cfg.env.dim_long_history_latent

        action_clamp = self.env.cfg.normalization.clip_actions
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        print(self.cfg["policy_class_name"])
        print(self.policy_cfg)

        if self.env.cfg.env.include_estimated_velocity_in_obs:
            # If the environment includes estimated velocity in observations, 
            # the number of proprioceptive observations is used directly.
            actor_critic: ActorCritic = actor_critic_class( self.env.dim_augmented_obs,
                                                            num_critic_obs,
                                                            self.env.num_actions,
                                                            num_long_history_length,
                                                            num_short_history_length,
                                                            dim_long_history_latent,
                                                            **self.policy_cfg).to(self.device)

        else: 
            actor_critic: ActorCritic = actor_critic_class( self.env.dim_proprio_obs,
                                                            num_critic_obs,
                                                            self.env.num_actions,
                                                            num_long_history_length,
                                                            num_short_history_length,
                                                            dim_long_history_latent,
                                                            **self.policy_cfg).to(self.device)
            
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        runner_device = self.device
        # Get the actual device of the estimator's first parameter
        # estimator_device = next(self.alg.actor_critic.actor.estimator.parameters()).device
        env_device = self.env.device
        
        print(f"Runner's configured device:    {runner_device}")
        print(f"Environment's reported device: {env_device}")
        # print(f"Estimator's actual device:     {estimator_device}")

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        # ******************** NEW MODIFICATION START ********************
        #
        # Link the velocity estimator from the policy to the environment.
        # This allows the environment to use the most up-to-date estimator
        # for generating observations at each timestep.
        if self.train_velestimator and hasattr(self.alg.actor_critic, 'actor') and hasattr(self.alg.actor_critic.actor, 'estimator'):
            print("Successfully linking velocity estimator from ActorCritic to environment.")
            # Assumes your environment class has the `set_velocity_estimator` method
            self.env.set_velocity_estimator(self.alg.actor_critic.actor.estimator)
        else:
            print("[Warning] OnPolicyRunner: Velocity estimator training is enabled, but no estimator was found in the ActorCritic model to link to the environment.")
    
    # ********************* NEW MODIFICATION END *********************
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Start
        # │
        # ├── Initialize Tensorboard, Observations, Buffers
        # │
        # └── Outer Loop (Training Iterations)
        #     │
        #     ├── Inner Loop (Rollout Steps)
        #     │   │
        #     │   ├── Get Actions → Env Step → Store Data
        #     │   │
        #     │   └── If Episode Done → Update Buffers
        #     │
        #     ├── Compute Returns & Advantages
        #     ├── Update Policy
        #     ├── Log Metrics
        #     └── If Save Interval → Save Model
        # │
        # └── Save Final Model
        # │
        # End
        # num_learning_iterations: Total number of training iterations to run

        # Initialization---------------------------------------------------------
        # initialize Tansorboard SummaryWriter
        os.makedirs(self.log_dir, exist_ok=True)

        if self.log_dir is not None and self.writer is None:
            # self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            self.writer = WandbSummaryWriter(log_dir=self.log_dir, cfg=self.cfg)
            self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        # Rollout Collection ----------------------------------------------------
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        # Double-Ended Queue
        # rewbuffer,lenbuffer: Buffers to track recent rewards and episode lengths, for logging and monitoring
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # Outer Loop (Training Iterations)
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Inner Loop (Rollout Steps)
            with torch.inference_mode():
                # self.num_steps_per_env: Number of steps to collect in the environment before updating
                for i in range(self.num_steps_per_env):
                    # get actions from the policy
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, _, _ = self.env.step(actions)

                    if torch.isnan(obs).any():
                        print("[NaN] in input observation to actor!")
                        print(f"obs: {obs}")

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # store data in RolloutStorage
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    # Tracks cumulative rewards (cur_reward_sum) and lengths (cur_episode_length).
                    # Resets these values for completed episodes (dones).
                    # only for monitoring and logging
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        # Tracks the total rewards for each environment in the current episode
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        # .extend() method appends elements of an iterable to the right end of the list.
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                # Compute Returns and Update---------------------------------------
                # Uses self.alg.compute_returns(critic_obs) to calculate:
                    # Returns (R_t): Total future rewards.
                    # Advantages (A_t): Difference between actual and predicted value.
                self.alg.compute_returns(critic_obs)
            
            # Policy and Value Update:
            # mean_value_loss and mean_surrogate_loss: Metrics to evaluate the performance of the PPO algorithm
            if self.train_velestimator:
                mean_value_loss, mean_surrogate_loss, ratio_stats, mean_estimation_loss,kl_logs  = self.alg.update()
            else:
                # mean_value_loss, mean_surrogate_loss, ratio_stats,kl_logs = self.alg.update()
                mean_value_loss, mean_surrogate_loss, ratio_stats = self.alg.update()
            if math.isnan(mean_value_loss) or math.isnan(mean_surrogate_loss):
                print(f"[NaN Warning] Loss contains NaN at iteration {it}")
                # Optionally stop training:
                # raise ValueError(f"NaN loss detected at iteration {it}")

            stop = time.time()
            learn_time = stop - start

            # Logging and Saving-------------------------------------------------
            if self.log_dir is not None:
                # The locals() function in Python is a built-in function 
                # that returns a dictionary representing the local symbol table
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        # ep_string = f''
        # if locs['ep_infos']:
        #     # The ep_infos list: infos['episode'] -> self.extras["episode"] in legged_robot.py
        #     for key in locs['ep_infos'][0]:
        #         infotensor = torch.tensor([], device=self.device)
        #         for ep_info in locs['ep_infos']:
        #             # handle scalar and zero dimensional tensor infos
        #             if not isinstance(ep_info[key], torch.Tensor):
        #                 ep_info[key] = torch.Tensor([ep_info[key]])
        #             if len(ep_info[key].shape) == 0:
        #                 ep_info[key] = ep_info[key].unsqueeze(0)
        #             infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        #         value = torch.mean(infotensor)
        #         self.writer.add_scalar('Episode/' + key, value, locs['it'])
        #         ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

                # safe_add_scalar(self.writer, 'Episode/' + key, value, locs['it'])
                # ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        if torch.isnan(self.alg.actor_critic.std).any() or torch.isinf(self.alg.actor_critic.std).any():
            print("[NaN/Inf] std parameter in ActorCritic is corrupted!")
            raise ValueError("std is invalid")


        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # if 'ratio_stats' in locs:
        #     # print("\nDebug: Ratio stats found:")
        #     for key, value in locs['ratio_stats'].items():
        #         self.writer.add_scalar(f'Policy/ratio_{key}', value, locs['it'])
        if 'ratio_stats' in locs:
            for key, value in locs['ratio_stats'].items():
                # safe_add_scalar(self.writer, f'Policy/ratio_{key}', value, locs['it'])
                self.writer.add_scalar(f'Policy/ratio_{key}', value, locs["it"])
        if 'kl_logs' in locs:
            for key, value in locs['kl_logs'].items():
                if value is not None:
                    # safe_add_scalar(self.writer, f'PPO/{key}', value, locs['it'])
                    self.writer.add_scalar(f'PPO/{key}', value, locs["it"])


         # Log transformer latent statistics
        # if hasattr(self.alg.actor_critic.actor, 'latent_stats'):
        #     stats = self.alg.actor_critic.actor.latent_stats
        #     for key, value in stats.items():
        #         self.writer.add_scalar(f'Transformer/latent_{key}', value, locs['it'])

        if hasattr(self.alg.actor_critic.actor, 'latent_stats'):
            stats = self.alg.actor_critic.actor.latent_stats
            for key, value in stats.items():
                # safe_add_scalar(self.writer, f'Transformer/latent_{key}', value, locs['it'])
                self.writer.add_scalar(f'Transformer/latent_{key}', value, locs["it"])


        # self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        # self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        # self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        # self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        # self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        # self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        # self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        # safe_add_scalar(self.writer, 'Loss/value_function', locs['mean_value_loss'], locs['it'])
        # safe_add_scalar(self.writer, 'Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        # safe_add_scalar(self.writer, 'Loss/learning_rate', self.alg.learning_rate, locs['it'])
        # safe_add_scalar(self.writer, 'Policy/mean_noise_std', mean_std.item(), locs['it'])
        # safe_add_scalar(self.writer, 'Perf/total_fps', fps, locs['it'])
        # safe_add_scalar(self.writer, 'Perf/collection time', locs['collection_time'], locs['it'])
        # safe_add_scalar(self.writer, 'Perf/learning_time', locs['learn_time'], locs['it'])

        # -- Losses
        # for key, value in locs["loss_dict"].items():
        #     self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        if self.train_velestimator:
            self.writer.add_scalar('Loss/Estimation Loss', locs['mean_estimation_loss'], locs['it'])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # if len(locs['rewbuffer']) > 0:
        #     self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
        #     self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
        #     self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
        #     self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        # if len(locs['rewbuffer']) > 0:
        #     mean_rew = statistics.mean(locs['rewbuffer'])
        #     mean_len = statistics.mean(locs['lenbuffer'])
        #     safe_add_scalar(self.writer, 'Train/mean_reward', mean_rew, locs['it'])
        #     safe_add_scalar(self.writer, 'Train/mean_episode_length', mean_len, locs['it'])
        #     safe_add_scalar(self.writer, 'Train/mean_reward/time', mean_rew, self.tot_time)
        #     safe_add_scalar(self.writer, 'Train/mean_episode_length/time', mean_len, self.tot_time)

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            # if self.alg.rnd:
            #     self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
            #     self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
            #     self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            # if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
            #     self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            #     self.writer.add_scalar(
            #         "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
            #     )

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Estimation loss:':>{pad}} {locs['mean_estimation_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        curr_it = locs['it']
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        hours = eta // 3600
        mins = int(eta // 60) % 60
        # mins = eta // 60
        # hours = mins // 60
        secs = eta % 60
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {hours:.0f}hours {mins:.0f}mins {secs:.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
  
    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.actor
    
    def get_actorcritic_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
