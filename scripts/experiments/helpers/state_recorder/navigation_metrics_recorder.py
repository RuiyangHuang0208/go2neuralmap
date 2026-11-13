import json
import math
import os
from copy import deepcopy

import torch
from loguru import logger

from .sub_recorder import SubRecorder


class NavigationMetricsRecorder(SubRecorder):
    """Record episode-level navigation metrics for the rollout starter path."""

    def __init__(self, **kwargs):
        self.logger = logger.bind(module_name="StateRecorder.NavMetrics")
        self.episodes = []
        self.current_episode = self._new_episode()

    def handle(self, var_dict):
        vel_cmd_info = var_dict.get("vel_cmd_generator_info")
        if (vel_cmd_info is None) or (len(vel_cmd_info.keys()) == 0):
            self.logger.warning("Got no navigation information")
            return

        current_xy = self._as_xy_tensor(vel_cmd_info.get("current_xy"))
        goal_xy = self._as_xy_tensor(vel_cmd_info.get("goal_xy"))
        if current_xy is None:
            self.logger.warning("Got no current_xy in navigation information")
            return

        episode = self.current_episode
        if episode["step_count"] == 0:
            episode["start_xy"] = current_xy.clone()
            episode["goal_xy"] = goal_xy.clone() if goal_xy is not None else None
            episode["last_xy"] = current_xy.clone()

        previous_xy = episode["last_xy"]
        if previous_xy is not None:
            episode["path_length"] += float(torch.norm(current_xy - previous_xy).item())
        episode["last_xy"] = current_xy.clone()

        episode["step_count"] += 1
        episode["elapsed_time_s"] += self._step_time(var_dict)
        self._accumulate_collisions(episode, var_dict.get("collision_flags"))

        task_terminal = bool(vel_cmd_info.get("task_terminal_flag", False))
        task_success = bool(vel_cmd_info.get("task_success_flag", False))
        task_terminal_reason = str(vel_cmd_info.get("task_terminal_reason", "none"))
        task_duration_s = self._optional_float(vel_cmd_info.get("task_elapsed_time_s"))
        reached_goal = bool(vel_cmd_info.get("reached_goal_flag", False))
        reset_buf = var_dict.get("reset_buf")
        done = bool(reset_buf[0].item()) if isinstance(reset_buf, torch.Tensor) and reset_buf.numel() > 0 else False

        if task_terminal:
            self._finish_episode(
                success=task_success,
                terminal_reason=task_terminal_reason,
                task_duration_s=task_duration_s,
            )
        elif reached_goal or done:
            self._finish_episode(success=reached_goal, terminal_reason="success" if reached_goal else "done")

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        summary = self._summary()
        payload = {
            "episodes": deepcopy(self.episodes),
            "current_episode": self._serializable_episode(self.current_episode),
            "summary": summary,
        }
        with open(os.path.join(save_dir, "navigation_metrics.json"), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        with open(os.path.join(save_dir, "navigation_metrics.jsonl"), "w") as f:
            for episode in self.episodes:
                f.write(json.dumps(episode, sort_keys=True) + "\n")

    def _finish_episode(self, *, success, terminal_reason, task_duration_s=None):
        episode = self.current_episode
        if episode["step_count"] <= 0:
            self.current_episode = self._new_episode()
            return

        start_xy = episode["start_xy"]
        goal_xy = episode["goal_xy"]
        shortest_path_length = 0.0
        if start_xy is not None and goal_xy is not None:
            shortest_path_length = float(torch.norm(goal_xy - start_xy).item())

        path_length = float(episode["path_length"])
        if success:
            denominator = max(path_length, shortest_path_length, 1.0e-8)
            success_per_length = shortest_path_length / denominator
        else:
            success_per_length = 0.0

        record = self._serializable_episode(episode)
        record.update(
            {
                "success": bool(success),
                "terminal_reason": terminal_reason,
                "shortest_path_length": shortest_path_length,
                "success_per_length": float(success_per_length),
                "time_to_goal_s": float(episode["elapsed_time_s"]) if success else None,
                "task_duration_s": (
                    float(task_duration_s)
                    if task_duration_s is not None
                    else float(episode["elapsed_time_s"])
                ),
            }
        )
        self.episodes.append(record)
        self.current_episode = self._new_episode()

    def _accumulate_collisions(self, episode, collision_flags):
        if not isinstance(collision_flags, torch.Tensor) or collision_flags.numel() == 0:
            return
        flags = collision_flags.detach().cpu().bool()
        if flags.dim() == 0:
            env0_flags = flags.view(1)
        elif flags.dim() == 1:
            env0_flags = flags
        else:
            env0_flags = flags[0].flatten()
        collision_step = bool(torch.any(env0_flags).item())
        if collision_step:
            episode["collision_count"] += 1
        if len(episode["directional_collision_counts"]) != int(env0_flags.numel()):
            episode["directional_collision_counts"] = [0 for _ in range(int(env0_flags.numel()))]
        for idx, flag in enumerate(env0_flags.tolist()):
            if flag:
                episode["directional_collision_counts"][idx] += 1

    def _summary(self):
        episode_count = len(self.episodes)
        if episode_count == 0:
            return {
                "episode_count": 0,
                "success_rate": None,
                "time_to_goal_mean_s": None,
                "collision_count_mean": None,
                "success_per_length_mean": None,
                "task_duration_mean_s": None,
                "timeout_count": 0,
                "done_count": 0,
                "timeout_rate": None,
                "done_rate": None,
            }

        success_values = [1.0 if episode["success"] else 0.0 for episode in self.episodes]
        timeout_values = [1.0 if episode["terminal_reason"] == "timeout" else 0.0 for episode in self.episodes]
        done_values = [1.0 if episode["terminal_reason"] == "done" else 0.0 for episode in self.episodes]
        successful_times = [
            float(episode["time_to_goal_s"])
            for episode in self.episodes
            if episode["time_to_goal_s"] is not None
        ]
        collision_counts = [float(episode["collision_count"]) for episode in self.episodes]
        spl_values = [float(episode["success_per_length"]) for episode in self.episodes]
        path_lengths = [float(episode["path_length"]) for episode in self.episodes]
        task_durations = [float(episode["task_duration_s"]) for episode in self.episodes]
        return {
            "episode_count": episode_count,
            "success_count": int(sum(success_values)),
            "timeout_count": int(sum(timeout_values)),
            "done_count": int(sum(done_values)),
            "success_rate": self._mean(success_values),
            "timeout_rate": self._mean(timeout_values),
            "done_rate": self._mean(done_values),
            "time_to_goal_mean_s": self._mean(successful_times),
            "task_duration_mean_s": self._mean(task_durations),
            "collision_count_mean": self._mean(collision_counts),
            "path_length_mean": self._mean(path_lengths),
            "success_per_length_mean": self._mean(spl_values),
        }

    @staticmethod
    def _new_episode():
        return {
            "step_count": 0,
            "elapsed_time_s": 0.0,
            "start_xy": None,
            "goal_xy": None,
            "last_xy": None,
            "path_length": 0.0,
            "collision_count": 0,
            "directional_collision_counts": [],
        }

    @staticmethod
    def _serializable_episode(episode):
        record = deepcopy(episode)
        for key in ("start_xy", "goal_xy", "last_xy"):
            value = record.get(key)
            if isinstance(value, torch.Tensor):
                record[key] = [float(item) for item in value.detach().cpu().flatten().tolist()]
        return record

    @staticmethod
    def _as_xy_tensor(value):
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().flatten()
        if tensor.numel() < 2:
            return None
        return tensor[:2].clone()

    @staticmethod
    def _step_time(var_dict):
        for key in ("dt", "step_dt", "step_time_s"):
            value = var_dict.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                return value
        return 0.0

    @staticmethod
    def _mean(values):
        if len(values) == 0:
            return None
        return float(sum(values) / len(values))

    @staticmethod
    def _optional_float(value):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value
