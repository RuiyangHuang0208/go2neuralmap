from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class NavigationTaskEvaluatorConfig:
    goal_tolerance: Optional[float] = 0.35
    goal_source: str = "sampled"
    goal_xy_list: Optional[List[List[float]]] = None
    goal_list_start_index: int = 0
    timeout_limit_s: Optional[float] = None
    timeout_limit_steps: Optional[int] = None
    timeout_action: str = "task_reset"
    advance_goal_on_success: bool = True
    advance_goal_on_timeout: bool = True
    advance_goal_on_done: bool = True


@dataclass
class NavigationTaskUpdate:
    reached_goal: torch.Tensor
    timeout: torch.Tensor
    terminal: torch.Tensor
    success_ids: torch.Tensor
    timeout_ids: torch.Tensor
    terminal_ids: torch.Tensor
    reset_env_ids: torch.Tensor
    task_reset_ids: torch.Tensor
    terminal_reason_by_env: List[str]
    info_by_env: List[Dict[str, Any]]


class NavigationTaskEvaluator:
    """Shared per-env task bookkeeping for navigation command generators."""

    def __init__(
        self,
        *,
        env,
        cfg: NavigationTaskEvaluatorConfig,
        default_goal_tolerance: Optional[float] = None,
    ):
        if cfg.goal_source not in ("sampled", "user", "list"):
            raise ValueError(f"Unknown goal_source: {cfg.goal_source}")
        if cfg.timeout_action not in ("task_reset", "env_reset"):
            raise ValueError(f"Unknown timeout_action: {cfg.timeout_action}")
        self.env = env
        self.cfg = cfg
        self.default_goal_tolerance = default_goal_tolerance
        self.device = env.device
        self.num_envs = int(env.num_envs)
        self.goal_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.user_goal_xy: Optional[torch.Tensor] = None
        self.goal_xy_list = self._validate_goal_xy_list(cfg.goal_xy_list)
        if cfg.goal_source == "list" and self.goal_xy_list is None:
            raise ValueError("goal_xy_list must be set when goal_source='list'")
        self.goal_list_index_by_env = torch.full(
            (self.num_envs,),
            int(cfg.goal_list_start_index),
            dtype=torch.long,
            device=self.device,
        )
        self.goal_index_by_env = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.reached_goal_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_elapsed_time_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.task_start_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

    def init_tasks(self, current_xy: torch.Tensor, *, resample: bool = True) -> None:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.refresh_goals(env_ids, resample=resample)
        self.start_tasks(env_ids, current_xy=current_xy)

    def set_goal_xy(self, goal_xy: torch.Tensor) -> None:
        goal_xy = torch.as_tensor(goal_xy, dtype=torch.float32, device=self.device)
        if goal_xy.dim() == 1:
            goal_xy = goal_xy.view(1, 2).expand(self.num_envs, -1)
        if goal_xy.shape != (self.num_envs, 2):
            raise ValueError(f"goal_xy must have shape ({self.num_envs}, 2), got {tuple(goal_xy.shape)}")
        self.user_goal_xy = goal_xy.clone()
        self.goal_xy.copy_(goal_xy)
        self.goal_index_by_env.fill_(-1)
        self._sync_env_task_goals(torch.arange(self.num_envs, device=self.device), self.goal_xy)

    def clear_user_goal(self, current_xy: torch.Tensor) -> None:
        self.user_goal_xy = None
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.refresh_goals(env_ids, resample=True)
        self.start_tasks(env_ids, current_xy=current_xy)

    def refresh_goals(self, env_ids: torch.Tensor, *, resample: bool) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        if self.user_goal_xy is not None:
            self.goal_xy[env_ids] = self.user_goal_xy[env_ids]
            self.goal_index_by_env[env_ids] = -1
            self._sync_env_task_goals(env_ids, self.goal_xy[env_ids])
            return
        if not resample:
            self._sync_env_task_goals(env_ids, self.goal_xy[env_ids])
            return
        if self.cfg.goal_source == "list":
            goals, goal_indices = self._next_goals_from_list(env_ids)
            self.goal_xy[env_ids] = goals
            self.goal_index_by_env[env_ids] = goal_indices
            self._sync_env_task_goals(env_ids, self.goal_xy[env_ids])
            return
        if hasattr(self.env, "resample_goals"):
            result = self.env.resample_goals(env_ids)
            assert result.goal_xy is not None
            sampled = result.goal_xy
        else:
            sampled = self.env.sample_goals(env_ids)[:, :2]
        self.goal_xy[env_ids] = sampled.to(device=self.device, dtype=torch.float32)
        self.goal_index_by_env[env_ids] = -1
        self._sync_env_task_goals(env_ids, self.goal_xy[env_ids])

    def start_tasks(self, env_ids: torch.Tensor, *, current_xy: torch.Tensor) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self.task_start_xy[env_ids] = current_xy[env_ids].to(device=self.device, dtype=torch.float32)
        self.task_elapsed_steps[env_ids] = 0
        self.task_elapsed_time_s[env_ids] = 0.0

    def handle_external_resets(self, reset_buf: Optional[torch.Tensor], current_xy: torch.Tensor) -> torch.Tensor:
        if reset_buf is None:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        reset_ids = reset_buf.to(device=self.device).bool().nonzero(as_tuple=False).flatten()
        if reset_ids.numel() == 0:
            return reset_ids
        self.refresh_goals(reset_ids, resample=bool(self.cfg.advance_goal_on_done))
        self.start_tasks(reset_ids, current_xy=current_xy)
        return reset_ids

    def update(
        self,
        *,
        current_xy: torch.Tensor,
        dt: float,
        reached_goal_override: Optional[torch.Tensor] = None,
    ) -> NavigationTaskUpdate:
        current_xy = current_xy.to(device=self.device, dtype=torch.float32)
        self.task_elapsed_steps += 1
        if dt >= 0.0:
            self.task_elapsed_time_s += float(dt)

        distance_to_goal = torch.norm(current_xy - self.goal_xy, dim=-1)
        if reached_goal_override is None:
            tolerance = self._goal_tolerance()
            reached_goal = distance_to_goal <= tolerance
        else:
            reached_goal = reached_goal_override.to(device=self.device).bool()

        timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.timeout_limit_s is not None:
            timeout |= self.task_elapsed_time_s >= float(self.cfg.timeout_limit_s)
        if self.cfg.timeout_limit_steps is not None:
            timeout |= self.task_elapsed_steps >= int(self.cfg.timeout_limit_steps)
        timeout &= ~reached_goal

        terminal = reached_goal | timeout
        success_ids = reached_goal.nonzero(as_tuple=False).flatten()
        timeout_ids = timeout.nonzero(as_tuple=False).flatten()
        terminal_ids = terminal.nonzero(as_tuple=False).flatten()
        if success_ids.numel() > 0:
            self.reached_goal_count[success_ids] += 1

        terminal_reason_by_env = ["none" for _ in range(self.num_envs)]
        for env_id in success_ids.tolist():
            terminal_reason_by_env[env_id] = "success"
        for env_id in timeout_ids.tolist():
            terminal_reason_by_env[env_id] = "timeout"

        reset_env_ids = (
            terminal_ids
            if terminal_ids.numel() > 0 and self.cfg.timeout_action == "env_reset" and timeout.any()
            else torch.zeros(0, dtype=torch.long, device=self.device)
        )
        task_reset_ids = terminal_ids
        info_by_env = self._build_info(
            current_xy=current_xy,
            distance_to_goal=distance_to_goal,
            reached_goal=reached_goal,
            timeout=timeout,
            terminal=terminal,
            terminal_reason_by_env=terminal_reason_by_env,
        )
        return NavigationTaskUpdate(
            reached_goal=reached_goal,
            timeout=timeout,
            terminal=terminal,
            success_ids=success_ids,
            timeout_ids=timeout_ids,
            terminal_ids=terminal_ids,
            reset_env_ids=reset_env_ids,
            task_reset_ids=task_reset_ids,
            terminal_reason_by_env=terminal_reason_by_env,
            info_by_env=info_by_env,
        )

    def update_env(
        self,
        *,
        env_id: int,
        current_xy: torch.Tensor,
        dt: float,
        reached_goal_override: Optional[bool] = None,
    ) -> NavigationTaskUpdate:
        env_id = int(env_id)
        current_xy_all = self.task_start_xy.clone()
        current_xy_all[env_id] = current_xy.to(device=self.device, dtype=torch.float32)
        self.task_elapsed_steps[env_id] += 1
        if dt >= 0.0:
            self.task_elapsed_time_s[env_id] += float(dt)

        distance_to_goal = torch.norm(current_xy_all - self.goal_xy, dim=-1)
        reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if reached_goal_override is None:
            reached_goal[env_id] = distance_to_goal[env_id] <= self._goal_tolerance()
        else:
            reached_goal[env_id] = bool(reached_goal_override)

        timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.timeout_limit_s is not None:
            timeout[env_id] |= self.task_elapsed_time_s[env_id] >= float(self.cfg.timeout_limit_s)
        if self.cfg.timeout_limit_steps is not None:
            timeout[env_id] |= self.task_elapsed_steps[env_id] >= int(self.cfg.timeout_limit_steps)
        timeout[env_id] &= ~reached_goal[env_id]

        terminal = reached_goal | timeout
        success_ids = reached_goal.nonzero(as_tuple=False).flatten()
        timeout_ids = timeout.nonzero(as_tuple=False).flatten()
        terminal_ids = terminal.nonzero(as_tuple=False).flatten()
        if reached_goal[env_id]:
            self.reached_goal_count[env_id] += 1

        terminal_reason_by_env = ["none" for _ in range(self.num_envs)]
        if reached_goal[env_id]:
            terminal_reason_by_env[env_id] = "success"
        if timeout[env_id]:
            terminal_reason_by_env[env_id] = "timeout"

        info_by_env = self._build_info(
            current_xy=current_xy_all,
            distance_to_goal=distance_to_goal,
            reached_goal=reached_goal,
            timeout=timeout,
            terminal=terminal,
            terminal_reason_by_env=terminal_reason_by_env,
        )
        return NavigationTaskUpdate(
            reached_goal=reached_goal,
            timeout=timeout,
            terminal=terminal,
            success_ids=success_ids,
            timeout_ids=timeout_ids,
            terminal_ids=terminal_ids,
            reset_env_ids=torch.zeros(0, dtype=torch.long, device=self.device),
            task_reset_ids=terminal_ids,
            terminal_reason_by_env=terminal_reason_by_env,
            info_by_env=info_by_env,
        )

    def complete_terminals(self, update: NavigationTaskUpdate, current_xy: torch.Tensor) -> None:
        if update.terminal_ids.numel() == 0:
            return
        success_ids = update.success_ids
        timeout_ids = update.timeout_ids
        if success_ids.numel() > 0:
            self.refresh_goals(success_ids, resample=bool(self.cfg.advance_goal_on_success))
        if timeout_ids.numel() > 0:
            self.refresh_goals(timeout_ids, resample=bool(self.cfg.advance_goal_on_timeout))
        self.start_tasks(update.terminal_ids, current_xy=current_xy)

    def info_for_env(self, update: NavigationTaskUpdate, env_id: int) -> Dict[str, Any]:
        return dict(update.info_by_env[env_id])

    def _build_info(
        self,
        *,
        current_xy: torch.Tensor,
        distance_to_goal: torch.Tensor,
        reached_goal: torch.Tensor,
        timeout: torch.Tensor,
        terminal: torch.Tensor,
        terminal_reason_by_env: List[str],
    ) -> List[Dict[str, Any]]:
        info_by_env = []
        for env_id in range(self.num_envs):
            info_by_env.append(
                {
                    "task_terminal_flag": bool(terminal[env_id].item()),
                    "task_success_flag": bool(reached_goal[env_id].item()),
                    "task_timeout_flag": bool(timeout[env_id].item()),
                    "task_terminal_reason": terminal_reason_by_env[env_id],
                    "task_elapsed_time_s": float(self.task_elapsed_time_s[env_id].item()),
                    "task_elapsed_steps": int(self.task_elapsed_steps[env_id].item()),
                    "task_start_xy": self.task_start_xy[env_id].detach().cpu(),
                    "current_xy": current_xy[env_id].detach().cpu(),
                    "goal_xy": self.goal_xy[env_id].detach().cpu(),
                    "distance_to_goal": distance_to_goal.detach().cpu(),
                    "goal_source": self.cfg.goal_source,
                    "goal_index": int(self.goal_index_by_env[env_id].item()),
                    "reached_goal_flag": bool(reached_goal[env_id].item()),
                    "reached_goal_count": int(self.reached_goal_count[env_id].item()),
                }
            )
        return info_by_env

    def _next_goals_from_list(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.goal_xy_list is None or self.goal_xy_list.numel() == 0:
            raise ValueError("goal_xy_list must contain at least one [x, y] goal")
        goal_count = int(self.goal_xy_list.shape[0])
        goal_indices = torch.remainder(self.goal_list_index_by_env[env_ids], goal_count)
        goals = self.goal_xy_list[goal_indices].to(device=self.device, dtype=torch.float32)
        self.goal_list_index_by_env[env_ids] = torch.remainder(goal_indices + 1, goal_count)
        return goals, goal_indices

    def _validate_goal_xy_list(self, goal_xy_list: Optional[List[List[float]]]) -> Optional[torch.Tensor]:
        if goal_xy_list is None:
            return None
        goals = torch.as_tensor(goal_xy_list, dtype=torch.float32, device=self.device)
        if goals.dim() != 2 or goals.shape[1] != 2 or goals.shape[0] == 0:
            raise ValueError(f"goal_xy_list must have shape (N, 2), got {tuple(goals.shape)}")
        return goals.clone()

    def _sync_env_task_goals(self, env_ids: torch.Tensor, goal_xy: torch.Tensor) -> None:
        if hasattr(self.env, "task_goal_xy"):
            self.env.task_goal_xy[env_ids] = goal_xy.to(device=self.device, dtype=self.env.task_goal_xy.dtype)
            if hasattr(self.env, "task_goal_valid"):
                self.env.task_goal_valid[env_ids] = True

    def _goal_tolerance(self) -> float:
        if self.cfg.goal_tolerance is not None:
            return float(self.cfg.goal_tolerance)
        if self.default_goal_tolerance is not None:
            return float(self.default_goal_tolerance)
        return 0.35

    def _normalize_env_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
