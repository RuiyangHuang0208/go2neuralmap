from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, List, Optional, Tuple, Dict
from legged_gym.envs.go2.go2_nm_env import GO2NM
import torch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import quat_apply
from loguru import logger

from utils.navigation import PlannerState, PlanResult

from .vel_command_generator import (VelCommandGenerator,
                                    VelCommandGeneratorConfig)
from .navigation_task_evaluator import NavigationTaskEvaluator, NavigationTaskEvaluatorConfig

@dataclass
class RunningInfo():
    current_xy: Optional[torch.Tensor] = None
    current_grid_xy: Optional[Tuple[int, int]] = None
    goal_xy: Optional[torch.Tensor] = None
    reached_goal_flag: bool = False
    terrain_reset_flag: bool = False
    reached_goal_count: int = 0
    traj_cost: float = 0.0


def draw_markers(
    env: GO2NM,
    current_xy: torch.Tensor,
    waypoints: Optional[List[torch.Tensor]],
    waypoint_idx: int,
    goal_xy: torch.Tensor,
):
    terrain_id = env.env_terrain_map[0]
    terrain_origin = env.terrain_loader._instances[terrain_id].array_offset

    robot_marker = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
    active_waypoint_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 0))
    waypoint_marker = gymutil.WireframeSphereGeometry(0.12, 4, 4, None, color=(1, 1, 0))
    goal_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 1))

    robot_h = env.terrain_loader.get_height(terrain_id, current_xy)
    goal_h = env.terrain_loader.get_height(terrain_id, goal_xy)

    robot_pose = gymapi.Transform(
        gymapi.Vec3(current_xy[0].item() + terrain_origin[0], current_xy[1].item() + terrain_origin[1], robot_h), r=None
    )
    goal_pose = gymapi.Transform(
        gymapi.Vec3(goal_xy[0].item() + terrain_origin[0], goal_xy[1].item() + terrain_origin[1], goal_h), r=None
    )

    # env.gym.clear_lines(env.viewer)
    # gymutil.draw_lines(robot_marker, env.gym, env.viewer, env.envs[0], robot_pose)
    # gymutil.draw_lines(goal_marker, env.gym, env.viewer, env.envs[0], goal_pose)
    markers = [
        (robot_marker, robot_pose),
        (goal_marker, goal_pose),
    ]

    if waypoints is not None:
        for idx, waypoint_xy in enumerate(waypoints):
            waypoint_h = env.terrain_loader.get_height(terrain_id, waypoint_xy)
            waypoint_pose = gymapi.Transform(
                gymapi.Vec3(
                    waypoint_xy[0].item() + terrain_origin[0],
                    waypoint_xy[1].item() + terrain_origin[1],
                    waypoint_h,
                ),
                r=None,
            )
            marker_style = active_waypoint_marker if idx == waypoint_idx else waypoint_marker
            markers.append((marker_style, waypoint_pose))

    return markers

@dataclass
class PositionNavigationConfig(VelCommandGeneratorConfig):
    replan_interval: int = 200
    goal_tolerance: Optional[float] = 0.35
    waypoint_tolerance: float = 0.25
    goal_source: str = "sampled"
    goal_xy_list: Optional[List[List[float]]] = None
    goal_list_start_index: int = 0
    resample_goal_flag: bool = False
    reset_env_flag: bool = True
    timeout_limit_s: Optional[float] = None
    timeout_limit_steps: Optional[int] = None
    timeout_action: str = "task_reset"
    advance_goal_on_success: bool = True
    advance_goal_on_timeout: bool = True
    advance_goal_on_done: bool = True
    terrain_reset_after_goal_count: int = 0
    terrain_reset_resample_goal: bool = True
    reset_maps_on_terrain_reset: bool = True
    map_least_count: float = 0.5
    planning_failure_callback: str = "keep_current"
    planning_failure_backward_velocity: float = 0.6
    planning_failure_replan_interval: int = 20
    pass


class NavigationMode(Enum):
    NORMAL = "normal"
    PLANNING_FAILURE_RECOVERY = "planning_failure_recovery"


class PositionNavigation(VelCommandGenerator):
    def __init__(self, cfg: PositionNavigationConfig,context=None, **kwargs):
        self.logger = logger.bind(module_name="VelCmdGen.PositionNavigation")
        if context is None:
            self.logger.error("No context is passed")
            raise ValueError("No context is passed")
        self.context = context
        self.env = self.context.env
        self.position_controller = self.context.position_controller
        self.markers = []

        self.cfg = cfg
        self.replan_interval = cfg.replan_interval
        self.resample_goal_flag = cfg.resample_goal_flag
        self.reset_env_flag = cfg.reset_env_flag
        task_cfg = NavigationTaskEvaluatorConfig(
            goal_tolerance=cfg.goal_tolerance,
            goal_source=cfg.goal_source,
            goal_xy_list=cfg.goal_xy_list,
            goal_list_start_index=cfg.goal_list_start_index,
            timeout_limit_s=cfg.timeout_limit_s,
            timeout_limit_steps=cfg.timeout_limit_steps,
            timeout_action=cfg.timeout_action,
            advance_goal_on_success=cfg.advance_goal_on_success,
            advance_goal_on_timeout=cfg.advance_goal_on_timeout,
            advance_goal_on_done=cfg.advance_goal_on_done,
        )
        self.task_evaluator = NavigationTaskEvaluator(env=self.env, cfg=task_cfg)

        self._init_per_env_state()
        self.info = {}
        pass

    def _init_per_env_state(self) -> None:
        num_envs = int(self.env.num_envs)
        self.goal_xy_by_env: List[Optional[torch.Tensor]] = [None for _ in range(num_envs)]
        self.waypoints_by_env: List[Optional[List[torch.Tensor]]] = [None for _ in range(num_envs)]
        self.waypoint_idx_by_env: List[int] = [0 for _ in range(num_envs)]
        self.traj_cost_by_env: List[float] = [0.0 for _ in range(num_envs)]
        self.last_grid_xy_by_env: List[Optional[Tuple[int, int]]] = [None for _ in range(num_envs)]
        self.navigation_mode_by_env: List[NavigationMode] = [NavigationMode.NORMAL for _ in range(num_envs)]
        self.recovery_step_count_by_env: List[int] = [0 for _ in range(num_envs)]
        self.step_count_by_env: List[int] = [0 for _ in range(num_envs)]
        self.reached_goal_count_by_env: List[int] = [0 for _ in range(num_envs)]
        self.info_by_env: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

    def _compute_current_heading(self, root_quat_xyzw: torch.Tensor) -> torch.Tensor:
        forward_vec = root_quat_xyzw.new_tensor([[1.0, 0.0, 0.0]])
        forward_dir = quat_apply(root_quat_xyzw.view(1, 4), forward_vec)
        return torch.atan2(forward_dir[:, 1], forward_dir[:, 0]).squeeze(0)

    def _get_current_heading_cmd(self, env_id: int) -> torch.Tensor:
        env_commands = getattr(self.env, "commands", None)
        if env_commands is not None and env_commands.shape[-1] > 3:
            return env_commands[env_id, 3].detach().clone()

        return self._compute_current_heading(self.env.root_states[env_id, 3:7])

    def _handle_planning_failure(self, env_id: int, current_xy: torch.Tensor) -> Optional[torch.Tensor]:
        callback_name = self.cfg.planning_failure_callback
        self.navigation_mode_by_env[env_id] = NavigationMode.PLANNING_FAILURE_RECOVERY
        self.recovery_step_count_by_env[env_id] = 0

        if callback_name == "keep_current":
            self.logger.warning("[Env {}][Replan] Planning failed, entering recovery mode with previous waypoints.", env_id)
            return None

        if callback_name == "backward_keep_heading_bypass_refiner":
            env_cmd = torch.zeros(4, dtype=torch.float, device=self.env.device)
            env_cmd[0] = -float(self.cfg.planning_failure_backward_velocity)
            env_cmd[3] = self._get_current_heading_cmd(env_id)
            self.logger.warning(
                "[Env {}][Replan] Planning failed, issuing backward fallback command and bypassing path refiner.",
                env_id,
            )
            return env_cmd

        raise ValueError(f"Unknown planning_failure_callback: {callback_name}")

    def _build_env_cmd(self, cmd_xy: torch.Tensor, heading_target: torch.Tensor) -> torch.Tensor:
        env_cmd = torch.zeros(4, dtype=torch.float, device=self.env.device)
        env_cmd[:2] = cmd_xy
        env_cmd[3] = heading_target
        return env_cmd

    def _reset_env(self, env_id: int) -> None:
        env_ids = torch.tensor([env_id], dtype=torch.long, device=self.env.device)
        if hasattr(self.env, "task_reset"):
            result = self.env.task_reset(env_ids, resample_goal=False)
            if result.goal_xy is not None:
                self.goal_xy_by_env[env_id] = result.goal_xy[0].to(device=self.env.device)
        else:
            self.env.reset_idx(env_ids)
        if hasattr(self.env, "terrain_loader") and hasattr(self.env, "env_terrain_map"):
            terrain_origins = self.env.terrain_loader.get_terrain_origin(self.env.env_terrain_map[env_ids])
            self.env.root_local_pos[env_ids] = self.env.root_states[env_ids, :3] - terrain_origins

    def _terrain_reset_env(self, env_id: int) -> None:
        if not hasattr(self.env, "terrain_reset"):
            raise AttributeError("PositionNavigation terrain reset requires env.terrain_reset")

        env_ids = torch.tensor([env_id], dtype=torch.long, device=self.env.device)
        result = self.env.terrain_reset(
            env_ids,
            resample_goal=bool(self.cfg.terrain_reset_resample_goal),
        )
        if result.goal_xy is not None:
            self.goal_xy_by_env[env_id] = result.goal_xy[0].to(device=self.env.device)
        else:
            self.goal_xy_by_env[env_id] = None
        if hasattr(self.env, "terrain_loader") and hasattr(self.env, "env_terrain_map"):
            terrain_origins = self.env.terrain_loader.get_terrain_origin(self.env.env_terrain_map[env_ids])
            self.env.root_local_pos[env_ids] = self.env.root_states[env_ids, :3] - terrain_origins

    def _reset_context_maps(self, env_id: int) -> None:
        reset_mask = torch.zeros(int(self.env.num_envs), dtype=torch.bool, device=self.env.device)
        reset_mask[env_id] = True
        if hasattr(self.context, "neural_map") and self.context.neural_map is not None:
            self.context.neural_map.reset(reset_mask)
        if hasattr(self.context, "cost_map") and self.context.cost_map is not None:
            self.context.cost_map.reset(reset_mask)

    def _should_terrain_reset(self, env_id: int) -> bool:
        threshold = int(self.cfg.terrain_reset_after_goal_count)
        return threshold > 0 and self.reached_goal_count_by_env[env_id] >= threshold

    def _attempt_replan_from_current_pose(self, env_id: int, current_xy: torch.Tensor) -> bool:
        goal_xy = self.goal_xy_by_env[env_id]
        if goal_xy is None:
            self._refresh_navigation_from_current_pose(env_id, current_xy, resample_goal=True)
            return self.waypoints_by_env[env_id] is not None

        new_waypoints = self._plan_waypoints(
            env_id=env_id,
            start_xy=current_xy,
            goal_xy=goal_xy,
            x_range=[-2.0, 10.0],
            y_range=[-2.0, 10.0],
            least_count=self.cfg.map_least_count,
            device=self.env.device,
        )

        if new_waypoints is None:
            return False

        self.waypoints_by_env[env_id] = new_waypoints
        self.waypoint_idx_by_env[env_id] = 0
        self.navigation_mode_by_env[env_id] = NavigationMode.NORMAL
        self.recovery_step_count_by_env[env_id] = 0
        self.logger.info("[Env {}][Replan] Planning recovered successfully. Returning to normal mode.", env_id)
        return True

    def _compute_controller_cmd(self, env_id: int, current_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        control_output = self.position_controller.compute_command(
            current_xy=current_xy,
            root_quat_xyzw=self.env.root_states[env_id, 3:7],
            target_xy=target_xy,
            waypoints=self.waypoints_by_env[env_id],
            waypoint_idx=self.waypoint_idx_by_env[env_id],
        )
        cmd_xy = control_output.command_xy
        heading_target = control_output.heading_target
        distance = control_output.distance
        self.logger.debug(f"Distance to sub target: {distance.item():.2f}")
        return self._build_env_cmd(cmd_xy, heading_target)

    def _compute_normal_mode_cmd(self, env_id: int, current_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        if self.step_count_by_env[env_id] > 0 and self.step_count_by_env[env_id] % self.replan_interval == 0:
            self.logger.info(
                "[Env {}][Replan] Step {}: replanning on updated cost map.",
                env_id,
                self.step_count_by_env[env_id],
            )
            if not self._attempt_replan_from_current_pose(env_id, current_xy):
                fallback_cmd = self._handle_planning_failure(env_id, current_xy)
                self.step_count_by_env[env_id] = 0
                if fallback_cmd is not None:
                    return fallback_cmd
            self.step_count_by_env[env_id] = 0

        self.step_count_by_env[env_id] += 1
        return self._compute_controller_cmd(env_id, current_xy, target_xy)

    def _compute_recovery_mode_cmd(self, env_id: int, current_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        self.recovery_step_count_by_env[env_id] += 1
        if self.recovery_step_count_by_env[env_id] % self.cfg.planning_failure_replan_interval == 0:
            self.logger.info(
                "[Env {}][Recovery] Step {}: retrying planning from current pose.",
                env_id,
                self.recovery_step_count_by_env[env_id],
            )
            if self._attempt_replan_from_current_pose(env_id, current_xy):
                self.step_count_by_env[env_id] = 1
                return self._compute_controller_cmd(env_id, current_xy, target_xy)

        if self.cfg.planning_failure_callback == "keep_current":
            self.step_count_by_env[env_id] += 1
            return self._compute_controller_cmd(env_id, current_xy, target_xy)

        if self.cfg.planning_failure_callback == "backward_keep_heading_bypass_refiner":
            env_cmd = torch.zeros(4, dtype=torch.float, device=self.env.device)
            env_cmd[0] = -float(self.cfg.planning_failure_backward_velocity)
            env_cmd[3] = self._get_current_heading_cmd(env_id)
            return env_cmd

        raise ValueError(f"Unknown planning_failure_callback: {self.cfg.planning_failure_callback}")

    ## Helper Functions
    def _xy_to_grid(
            self,
            xy: torch.Tensor,
            x_range: Tuple[float, float],
            y_range: Tuple[float, float],
            least_count: float) -> Tuple[int, int]:
        """Convert world xy to cost-map grid (row=x_idx, col=y_idx)."""
        x_idx = int(torch.floor((xy[0] - x_range[0]) / least_count).item())
        y_idx = int(torch.floor((xy[1] - y_range[0]) / least_count).item())
        return (x_idx, y_idx)

    def _grid_to_xy(
        self,
        grid: Tuple[int, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        least_count: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert cost-map grid (row, col) back to world xy (cell center)."""
        x = x_range[0] + (grid[0] + 0.5) * least_count
        y = y_range[0] + (grid[1] + 0.5) * least_count
        return torch.tensor([x, y], dtype=torch.float, device=device)

    def _clamp_grid(
            self,
            grid: Tuple[int, int],
            h: int, w: int) -> Tuple[int, int]:
        return (max(0, min(grid[0], h - 1)), max(0, min(grid[1], w - 1)))

    def _is_cardinal_adjacent_or_same(
        self,
        prev_grid: Tuple[int, int],
        curr_grid: Tuple[int, int],
    ) -> bool:
        delta_x = curr_grid[0] - prev_grid[0]
        delta_y = curr_grid[1] - prev_grid[1]
        return (
            (delta_x == 0 and delta_y == 0)
            or (abs(delta_x) == 1 and delta_y == 0)
            or (delta_x == 0 and abs(delta_y) == 1)
        )

    def _refresh_navigation_from_current_pose(
        self,
        env_id: int,
        current_xy: torch.Tensor,
        *,
        resample_goal: bool,
    ) -> None:
        self.last_grid_xy_by_env[env_id] = self._xy_to_grid(
            current_xy,
            x_range=[-2.0, 10.0],
            y_range=[-2.0, 10.0],
            least_count=self.cfg.map_least_count,
        )
        self.traj_cost_by_env[env_id] = 0.0
        self.step_count_by_env[env_id] = 0

        effective_resample_goal = bool(resample_goal or self.goal_xy_by_env[env_id] is None)
        if effective_resample_goal:
            env_ids = torch.tensor([env_id], dtype=torch.long, device=self.env.device)
            self.task_evaluator.refresh_goals(env_ids, resample=True)
            self.goal_xy_by_env[env_id] = self.task_evaluator.goal_xy[env_id].detach().clone()
            self.logger.info("[Env {}] Sampled goal at: {}", env_id, self.goal_xy_by_env[env_id])
        elif self.goal_xy_by_env[env_id] is not None:
            self.task_evaluator.goal_xy[env_id] = self.goal_xy_by_env[env_id].to(
                device=self.env.device,
                dtype=torch.float32,
            )
            self.task_evaluator.refresh_goals(
                torch.tensor([env_id], dtype=torch.long, device=self.env.device),
                resample=False,
            )
        self.task_evaluator.start_tasks(
            torch.tensor([env_id], dtype=torch.long, device=self.env.device),
            current_xy=self.env.root_local_pos[:, :2],
        )

        self.waypoints_by_env[env_id] = self._plan_waypoints(
            env_id=env_id,
            start_xy=current_xy,
            goal_xy=self.goal_xy_by_env[env_id],
            x_range=[-2.0, 10.0],
            y_range=[-2.0, 10.0],
            least_count=self.cfg.map_least_count,
            device=self.env.device
        )
        self.waypoint_idx_by_env[env_id] = 0

    def _plan_waypoints(
        self,
        env_id: int,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        least_count: float,
        device: torch.device,
    ) -> Optional[List[torch.Tensor]]:
        """Plan a path from start_xy to goal_xy using A* on the current cost map.

        Returns a list of world-xy waypoint tensors (subsampled), or None on failure.
        """
        planner_map = self.context.cost_map.get_planner_map(env_id=env_id).detach().cpu()
        h, w = planner_map.shape[:2]

        start_grid = self._clamp_grid(
            self._xy_to_grid(start_xy.cpu(), x_range, y_range, least_count),
            h, w)
        goal_grid = self._clamp_grid(
            self._xy_to_grid(goal_xy.cpu(), x_range, y_range, least_count),
            h, w)

        result = self.context.planner.plan(start_grid, goal_grid, planner_map)

        if result.state != PlannerState.SUCCESS or result.path is None:
            self.logger.warning(
                "[Env {}][Planner] Failed: {} | start_grid={} goal_grid={}",
                env_id,
                result.state.name,
                start_grid,
                goal_grid,
            )
            return None

        path = result.path
        self.logger.success("[Env {}][Planner] Success: {} steps, cost={:.2f}", env_id, len(path), result.total_cost)

        # Subsample path waypoints (always include last)
        # indices = list(range(0, len(path), WAYPOINT_STRIDE))
        # if indices[-1] != len(path) - 1:
        #     indices.append(len(path) - 1)
        indices = self.context.planner.subsample_waypoints(result)

        waypoints = [
            self._grid_to_xy(path[i], x_range, y_range, least_count, device)
            for i in indices
        ]
        waypoints[-1] = goal_xy
        if len(waypoints) > 1:
            del waypoints[0]
        return waypoints

    def retrieve_markers(self):
        markers = self.markers
        return markers

    def init_state(self):
        if len(self.goal_xy_by_env) != int(self.env.num_envs):
            self._init_per_env_state()
        for env_id in range(int(self.env.num_envs)):
            current_xy = self.env.root_local_pos[env_id, :2]
            self._refresh_navigation_from_current_pose(env_id, current_xy, resample_goal=True)
            self.navigation_mode_by_env[env_id] = NavigationMode.NORMAL
            self.recovery_step_count_by_env[env_id] = 0

    def compute_vel_cmd(self, **kwargs):
        commands = torch.zeros((int(self.env.num_envs), 4), dtype=torch.float, device=self.env.device)
        self.info_by_env = []
        for env_id in range(int(self.env.num_envs)):
            commands[env_id] = self._compute_vel_cmd_for_env(env_id)
        self.info = dict(self.info_by_env[0]) if self.info_by_env else {}
        self.info["per_env"] = self.info_by_env
        return commands

    def _compute_vel_cmd_for_env(self, env_id: int) -> torch.Tensor:
        info_struct = RunningInfo()
        current_xy = self.env.root_local_pos[env_id, :2]
        current_grid_xy = self._xy_to_grid(
            current_xy,
            x_range=[-2.0, 10.0],
            y_range=[-2.0, 10.0],
            least_count=self.cfg.map_least_count,
        )
        info_struct.current_xy = current_xy.detach().cpu()
        info_struct.current_grid_xy = current_grid_xy

        reset_buf = getattr(self.env, "reset_buf", None)
        if reset_buf is not None and bool(reset_buf[env_id].item()):
            self.logger.info("[Env {}][Reset] Environment reset detected. Reinitializing navigation state.", env_id)
            self._refresh_navigation_from_current_pose(env_id, current_xy, resample_goal=True)
            self.navigation_mode_by_env[env_id] = NavigationMode.NORMAL
            self.recovery_step_count_by_env[env_id] = 0
            current_xy = self.env.root_local_pos[env_id, :2]
            current_grid_xy = self._xy_to_grid(
                current_xy,
                x_range=[-2.0, 10.0],
                y_range=[-2.0, 10.0],
                least_count=self.cfg.map_least_count,
            )
            info_struct.current_xy = current_xy.detach().cpu()
            info_struct.current_grid_xy = current_grid_xy

        last_grid_xy = self.last_grid_xy_by_env[env_id]
        if last_grid_xy is None:
            self.last_grid_xy_by_env[env_id] = current_grid_xy
        elif not self._is_cardinal_adjacent_or_same(last_grid_xy, current_grid_xy):
            self.logger.warning(
                "[Env {}][Reset] Non-local grid jump detected: {} -> {}. Reanchoring navigation state.",
                env_id,
                last_grid_xy,
                current_grid_xy,
            )
            self._refresh_navigation_from_current_pose(env_id, current_xy, resample_goal=True)
            self.navigation_mode_by_env[env_id] = NavigationMode.NORMAL
            self.recovery_step_count_by_env[env_id] = 0
            current_grid_xy = self.last_grid_xy_by_env[env_id]
            info_struct.current_grid_xy = current_grid_xy
        elif last_grid_xy != current_grid_xy:
            self.traj_cost_by_env[env_id] += self.context.cost_map.get_traversal_cost(
                last_grid_xy,
                current_grid_xy,
                env_id=env_id,
            )
            self.last_grid_xy_by_env[env_id] = current_grid_xy

        info_struct.traj_cost = self.traj_cost_by_env[env_id]
        info_struct.reached_goal_count = self.reached_goal_count_by_env[env_id]

        # --- Advance waypoints ---
        waypoints = self.waypoints_by_env[env_id]
        waypoint_idx = self.waypoint_idx_by_env[env_id]
        reached_goal = waypoints is None or waypoint_idx >= len(waypoints)
        target_xy = current_xy
        if not reached_goal:
            target_xy = waypoints[waypoint_idx]
            dist_to_waypoint = torch.norm(current_xy - target_xy)
            if dist_to_waypoint <= float(self.cfg.waypoint_tolerance):
                self.waypoint_idx_by_env[env_id] += 1
                waypoint_idx = self.waypoint_idx_by_env[env_id]
                if waypoint_idx < len(waypoints):
                    self.logger.info(
                        "[Env {}] Reached waypoint {}, next={}",
                        env_id,
                        waypoint_idx - 1,
                        waypoints[waypoint_idx].tolist(),
                    )
                    target_xy = waypoints[waypoint_idx]
                else:
                    reached_goal = True

        task_update = self.task_evaluator.update_env(
            env_id=env_id,
            current_xy=current_xy,
            dt=float(getattr(self.env, "dt", 0.0)),
            reached_goal_override=bool(reached_goal),
        )
        timeout = bool(task_update.timeout[env_id].item())
        task_terminal = bool(task_update.terminal[env_id].item())

        if task_terminal:
            # --- Reached Goal Point ---
            if timeout:
                self.logger.info("[Env {}][Timeout] Traj Cost: {}", env_id, self.traj_cost_by_env[env_id])
            else:
                self.logger.info("[Env {}] Traj Cost: {}", env_id, self.traj_cost_by_env[env_id])
            self.traj_cost_by_env[env_id] = 0.
            if reached_goal:
                self.reached_goal_count_by_env[env_id] += 1
            info_struct.reached_goal_count = self.reached_goal_count_by_env[env_id]
            should_reset_env = (reached_goal and self.reset_env_flag) or (timeout and self.cfg.timeout_action == "env_reset")
            if should_reset_env:
                info_struct.reached_goal_flag = bool(reached_goal)
                if reached_goal and self._should_terrain_reset(env_id):
                    self.logger.info(
                        "[Env {}][TerrainReset] Reached goal count {} hit threshold {}. Resetting terrain.",
                        env_id,
                        self.reached_goal_count_by_env[env_id],
                        int(self.cfg.terrain_reset_after_goal_count),
                    )
                    info_struct.terrain_reset_flag = True
                    self._terrain_reset_env(env_id)
                    if self.cfg.reset_maps_on_terrain_reset:
                        self._reset_context_maps(env_id)
                    self.reached_goal_count_by_env[env_id] = 0
                    info_struct.reached_goal_count = 0
                    self.navigation_mode_by_env[env_id] = NavigationMode.NORMAL
                    self.recovery_step_count_by_env[env_id] = 0
                else:
                    self._reset_env(env_id)
                current_xy = self.env.root_local_pos[env_id, :2]
                refresh_resample_goal = (
                    bool(self.cfg.advance_goal_on_success)
                    if reached_goal
                    else bool(self.cfg.advance_goal_on_timeout)
                )
                if info_struct.terrain_reset_flag:
                    refresh_resample_goal = self.goal_xy_by_env[env_id] is None
                self._refresh_navigation_from_current_pose(
                    env_id,
                    current_xy,
                    resample_goal=refresh_resample_goal,
                )

            if not should_reset_env:
                refresh_resample_goal = (
                    bool(self.cfg.advance_goal_on_success)
                    if reached_goal
                    else bool(self.cfg.advance_goal_on_timeout)
                )
                self._refresh_navigation_from_current_pose(env_id, current_xy, resample_goal=refresh_resample_goal)
                self.navigation_mode_by_env[env_id] = NavigationMode.NORMAL
                self.recovery_step_count_by_env[env_id] = 0

            waypoints = self.waypoints_by_env[env_id]
            waypoint_idx = self.waypoint_idx_by_env[env_id]
            if waypoints is not None and waypoint_idx < len(waypoints):
                target_xy = waypoints[waypoint_idx]
            else:
                target_xy = current_xy # Stop at goal point

        if env_id == 0 and self.goal_xy_by_env[env_id] is not None:
            self.markers = draw_markers(
                env=self.env,
                current_xy=current_xy,
                waypoints=self.waypoints_by_env[env_id],
                waypoint_idx=self.waypoint_idx_by_env[env_id],
                goal_xy=self.goal_xy_by_env[env_id]
            )

        if self.navigation_mode_by_env[env_id] == NavigationMode.PLANNING_FAILURE_RECOVERY:
            env_cmd = self._compute_recovery_mode_cmd(env_id, current_xy, target_xy)
        else:
            env_cmd = self._compute_normal_mode_cmd(env_id, current_xy, target_xy)

        if self.goal_xy_by_env[env_id] is not None:
            info_struct.goal_xy = self.goal_xy_by_env[env_id].detach().cpu()
        info = asdict(info_struct)
        info.update(self.task_evaluator.info_for_env(task_update, env_id))
        info["env_id"] = env_id
        info["planning_failure_callback"] = self.cfg.planning_failure_callback
        info["navigation_mode"] = self.navigation_mode_by_env[env_id].value
        info["recovery_step_count"] = self.recovery_step_count_by_env[env_id]
        self.info_by_env.append(info)

        return env_cmd

    def get_info(self) -> Dict:
        return self.info
