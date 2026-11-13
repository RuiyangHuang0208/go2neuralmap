from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from isaacgym import gymapi, gymutil
import torch

from .navigation_task_evaluator import NavigationTaskEvaluator, NavigationTaskEvaluatorConfig
from .vel_command_generator import VelCommandGenerator, VelCommandGeneratorConfig


@dataclass
class PointToPointNavigationConfig(VelCommandGeneratorConfig):
    goal_tolerance: Optional[float] = 0.35
    goal_source: str = "sampled"
    goal_xy_list: Optional[List[List[float]]] = None
    goal_list_start_index: int = 0
    reset_env_flag: bool = False
    timeout_limit_s: Optional[float] = None
    timeout_limit_steps: Optional[int] = None
    timeout_action: str = "task_reset"
    advance_goal_on_success: bool = True
    advance_goal_on_timeout: bool = True
    advance_goal_on_done: bool = True
    draw_goal_marker: bool = True
    goal_marker_radius: float = 0.15


@dataclass
class PointToPointNavigationInfo:
    current_xy: Optional[torch.Tensor] = None
    current_grid_xy: Optional[Tuple[int, int]] = None
    goal_xy: Optional[torch.Tensor] = None
    distance_to_goal: Optional[torch.Tensor] = None
    reached_goal_flag: bool = False
    reached_goal_count: int = 0
    command: Optional[torch.Tensor] = None
    heading_target: Optional[torch.Tensor] = None


class PointToPointNavigation(VelCommandGenerator):
    def __init__(self, cfg: PointToPointNavigationConfig, context=None, **kwargs):
        if context is None:
            raise ValueError("No context is passed")
        self.cfg = cfg
        self.context = context
        self.env = context.env
        self.position_controller = context.position_controller
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
        self.goal_xy = self.task_evaluator.goal_xy
        self.markers = []
        self.info: Dict = {}

    def set_goal_xy(self, goal_xy: torch.Tensor) -> None:
        self.task_evaluator.set_goal_xy(goal_xy)

    def clear_user_goal(self) -> None:
        current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
        self.task_evaluator.clear_user_goal(current_xy)

    def init_state(self, **kwargs):
        current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
        self.task_evaluator.init_tasks(current_xy, resample=True)
        self.info = {}

    @torch.no_grad()
    def compute_vel_cmd(self, **kwargs):
        current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
        reset_buf = getattr(self.env, "reset_buf", None)
        self.task_evaluator.handle_external_resets(reset_buf, current_xy)

        task_update = self.task_evaluator.update(
            current_xy=current_xy,
            dt=float(getattr(self.env, "dt", 0.0)),
        )
        terminal_ids = task_update.terminal_ids
        if terminal_ids.numel() > 0:
            timeout_reset_ids = (
                task_update.timeout_ids
                if self.cfg.timeout_action == "env_reset"
                else torch.zeros(0, dtype=torch.long, device=self.env.device)
            )
            success_reset_ids = (
                task_update.success_ids
                if self.cfg.reset_env_flag
                else torch.zeros(0, dtype=torch.long, device=self.env.device)
            )
            reset_terminal_ids = torch.unique(torch.cat((timeout_reset_ids, success_reset_ids)))
            if reset_terminal_ids.numel() > 0:
                self._reset_envs(reset_terminal_ids)
                current_xy = self.env.root_local_pos[:, :2].to(device=self.env.device, dtype=torch.float32)
            self.task_evaluator.complete_terminals(task_update, current_xy=current_xy)

        env_cmd = torch.zeros_like(self.env.commands)
        command_xy_by_env = []
        heading_target_by_env = []
        current_grid_xy_by_env = []
        for env_id in range(int(self.env.num_envs)):
            control_output = self.position_controller.compute_command(
                current_xy=current_xy[env_id],
                root_quat_xyzw=self.env.root_states[env_id, 3:7],
                target_xy=self.goal_xy[env_id],
                waypoints=None,
                waypoint_idx=0,
            )
            env_cmd[env_id, :2] = control_output.command_xy.to(device=self.env.device, dtype=env_cmd.dtype)
            if env_cmd.shape[1] > 3:
                env_cmd[env_id, 3] = control_output.heading_target.to(device=self.env.device, dtype=env_cmd.dtype)
            elif env_cmd.shape[1] > 2 and hasattr(self.env, "yaw"):
                heading_error = self._wrap_to_pi(control_output.heading_target - self.env.yaw[env_id])
                env_cmd[env_id, 2] = torch.clamp(0.5 * heading_error, min=-1.0, max=1.0).to(dtype=env_cmd.dtype)
            command_xy_by_env.append(control_output.command_xy.detach().cpu())
            heading_target_by_env.append(control_output.heading_target.detach().cpu())
            current_grid_xy_by_env.append(self._current_grid_xy(current_xy[env_id]))

        self._update_markers()
        distance_to_goal = torch.norm(current_xy - self.goal_xy, dim=-1).detach().cpu()
        info_struct = PointToPointNavigationInfo(
            current_xy=current_xy[0].detach().cpu(),
            current_grid_xy=current_grid_xy_by_env[0],
            goal_xy=self.goal_xy[0].detach().cpu(),
            distance_to_goal=distance_to_goal,
            reached_goal_flag=bool(task_update.reached_goal[0].item()),
            reached_goal_count=int(self.task_evaluator.reached_goal_count[0].item()),
            command=command_xy_by_env[0],
            heading_target=heading_target_by_env[0],
        )
        self.info = asdict(info_struct)
        self.info.update(self.task_evaluator.info_for_env(task_update, 0))
        self.info["per_env"] = [
            {
                **self.task_evaluator.info_for_env(task_update, env_id),
                "command": command_xy_by_env[env_id],
                "heading_target": heading_target_by_env[env_id],
                "current_grid_xy": current_grid_xy_by_env[env_id],
            }
            for env_id in range(int(self.env.num_envs))
        ]
        return env_cmd

    def retrieve_markers(self) -> List:
        return self.markers

    def get_info(self) -> Dict:
        return self.info

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.env.device, dtype=torch.long)
        if hasattr(self.env, "task_reset"):
            self.env.task_reset(env_ids, resample_goal=False)
        else:
            self.env.reset_idx(env_ids)

    def _current_grid_xy(self, current_xy: torch.Tensor) -> Tuple[int, int]:
        neural_map = self.context.neural_map
        if not hasattr(neural_map, "maps"):
            return None
        cfg = neural_map.cfg
        x_idx = int(torch.floor((current_xy[0] - float(cfg.x_range[0])) / float(cfg.least_count)).item())
        y_idx = int(torch.floor((current_xy[1] - float(cfg.y_range[0])) / float(cfg.least_count)).item())
        x_length, y_length = neural_map.maps.shape[1:3]
        return (
            max(0, min(x_idx, int(x_length) - 1)),
            max(0, min(y_idx, int(y_length) - 1)),
        )

    def _update_markers(self) -> None:
        self.markers = []
        if not bool(self.cfg.draw_goal_marker) or self.env.num_envs <= 0:
            return
        if not hasattr(self.env, "terrain_loader") or not hasattr(self.env, "env_terrain_map"):
            return
        terrain_id = self.env.env_terrain_map[0]
        terrain_idx = int(terrain_id.detach().cpu().item()) if isinstance(terrain_id, torch.Tensor) else int(terrain_id)
        terrain_origin = self.env.terrain_loader._instances[terrain_idx].array_offset
        goal_h = self.env.terrain_loader.get_height(terrain_id, self.goal_xy[0])
        goal_marker = gymutil.WireframeSphereGeometry(
            float(self.cfg.goal_marker_radius),
            8,
            8,
            None,
            color=(0, 1, 1),
        )
        goal_pose = gymapi.Transform(
            gymapi.Vec3(
                self.goal_xy[0, 0].item() + terrain_origin[0],
                self.goal_xy[0, 1].item() + terrain_origin[1],
                goal_h,
            ),
            r=None,
        )
        self.markers.append((goal_marker, goal_pose))

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi
