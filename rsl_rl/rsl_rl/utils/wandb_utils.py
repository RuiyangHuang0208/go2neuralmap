# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from dataclasses import asdict

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("WandB is required to log to Weights and Biases.")

def convert_cfg_to_dict(cfg):
    if hasattr(cfg, "__dict__"):
        return {k: convert_cfg_to_dict(v) for k, v in cfg.__dict__.items() if not k.startswith('__')}
    elif isinstance(cfg, (list, tuple)):
        return [convert_cfg_to_dict(v) for v in cfg]
    elif isinstance(cfg, dict):
        return {k: convert_cfg_to_dict(v) for k, v in cfg.items()}
    else:
        return cfg

class WandbSummaryWriter:
    """WandB-only summary writer. Logs metrics, configs, and models using Weights & Biases."""

    def __init__(self, log_dir: str, cfg):
        self.log_dir = log_dir

        # Get the run name from config or the folder name.
        run_name = cfg.get("wandb_run_name", None)
        if run_name is None:
            run_name = os.path.split(log_dir)[-1]

        # Load project and entity from config/env
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify 'wandb_project' in the runner config.")

        entity = cfg.get("wandb_entity", None)
        if entity is None:
            entity = os.environ.get("WANDB_USERNAME", None)
        mode = cfg.get("wandb_mode", None)

        # Initialize wandb
        # wandb.login(key="9a8eac2e29b1a3f9de782fd285b2fb0c4dbe3952")
        init_kwargs = {"project": project, "entity": entity, "name": run_name}
        if mode is not None:
            init_kwargs["mode"] = mode
        wandb.init(**init_kwargs)
        wandb.config.update({"log_dir": log_dir})

        # Optional tag renaming
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }
        self._last_int_step = None

    def log_dict(self, metrics: dict, step=None):
        """Log a whole dictionary of metrics to WandB."""
        mapped = {self._map_path(k): v for k, v in metrics.items()}
        wandb.log(mapped, step=step)

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Log a scalar value to WandB."""
        mapped_tag = self._map_path(tag)
        payload = {mapped_tag: self._to_scalar(scalar_value)}
        step = self._normalize_step(global_step)
        if isinstance(global_step, float) and step is not None:
            payload["global_time"] = float(global_step)
        if step is not None:
            wandb.log(payload, step=step)
            return
        wandb.log(payload)

    def _to_scalar(self, scalar_value):
        if isinstance(scalar_value, float) or isinstance(scalar_value, int):
            return scalar_value
        try:
            return float(scalar_value)
        except Exception:
            print(f"[Warning] Failed to convert scalar value: {scalar_value}")
            return 0.0

    def _normalize_step(self, global_step):
        if isinstance(global_step, bool) or global_step is None:
            return None
        if isinstance(global_step, int):
            self._last_int_step = int(global_step)
            return int(global_step)
        if isinstance(global_step, float):
            return self._last_int_step
        try:
            step = int(global_step)
        except Exception:
            return None
        self._last_int_step = step
        return step

    def stop(self):
        """Finish the current WandB run."""
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Log full config to WandB."""
        wandb.config.update({
            "env_cfg": convert_cfg_to_dict(env_cfg),
            "runner_cfg": convert_cfg_to_dict(runner_cfg),
            "alg_cfg": convert_cfg_to_dict(alg_cfg),
            "policy_cfg": convert_cfg_to_dict(policy_cfg),
        })

    def save_model(self, model_path, iter):
        """Track a model checkpoint."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        """Track an arbitrary file."""
        wandb.save(path, base_path=os.path.dirname(path))

    def _map_path(self, path):
        """Optionally rename paths to cleaner tag names."""
        return self.name_map.get(path, path)

# from __future__ import annotations

# import os
# from dataclasses import asdict
# from torch.utils.tensorboard import SummaryWriter

# try:
#     import wandb
# except ModuleNotFoundError:
#     raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")

# def convert_cfg_to_dict(cfg):
#     if hasattr(cfg, "__dict__"):
#         return {k: convert_cfg_to_dict(v) for k, v in cfg.__dict__.items() if not k.startswith('__')}
#     elif isinstance(cfg, (list, tuple)):
#         return [convert_cfg_to_dict(v) for v in cfg]
#     elif isinstance(cfg, dict):
#         return {k: convert_cfg_to_dict(v) for k, v in cfg.items()}
#     else:
#         return cfg

# class WandbSummaryWriter(SummaryWriter):
#     """Summary writer for Weights and Biases."""

#     def __init__(self, log_dir: str, flush_secs: int, cfg):
#         super().__init__(log_dir, flush_secs)

#         # Get the run name
#         run_name = os.path.split(log_dir)[-1]

#         try:
#             project = cfg["wandb_project"]
#         except KeyError:
#             raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

#         try:
#             entity = os.environ["WANDB_USERNAME"]
#         except KeyError:
#             entity = None

#         # Initialize wandb
#         wandb.init(project=project, entity=entity, name=run_name)

#         # Add log directory to wandb
#         wandb.config.update({"log_dir": log_dir})

#         self.name_map = {
#             "Train/mean_reward/time": "Train/mean_reward_time",
#             "Train/mean_episode_length/time": "Train/mean_episode_length_time",
#         }




#     def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
#         # wandb.config.update({"runner_cfg": runner_cfg})
#         # wandb.config.update({"policy_cfg": policy_cfg})
#         # wandb.config.update({"alg_cfg": alg_cfg})
#         # try:
#         #     wandb.config.update({"env_cfg": env_cfg.to_dict()})
#         # except Exception:
#         #     wandb.config.update({"env_cfg": asdict(env_cfg)})
#         wandb.config.update({
#             "env_cfg": convert_cfg_to_dict(env_cfg),
#             "runner_cfg": convert_cfg_to_dict(runner_cfg),
#             "alg_cfg": convert_cfg_to_dict(alg_cfg),
#             "policy_cfg": convert_cfg_to_dict(policy_cfg),
#         })


#     def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
#         super().add_scalar(
#             tag,
#             scalar_value,
#             global_step=global_step,
#             walltime=walltime,
#             new_style=new_style,
#         )
#         wandb.log({self._map_path(tag): scalar_value}, step=global_step)

#     def stop(self):
#         wandb.finish()

#     def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
#         self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

#     def save_model(self, model_path, iter):
#         wandb.save(model_path, base_path=os.path.dirname(model_path))

#     def save_file(self, path, iter=None):
#         wandb.save(path, base_path=os.path.dirname(path))

#     """
#     Private methods.
#     """

#     def _map_path(self, path):
#         if path in self.name_map:
#             return self.name_map[path]
#         else:
#             return path
