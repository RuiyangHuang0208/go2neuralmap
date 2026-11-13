import os
from datetime import datetime
from typing import List, Optional, Tuple

import isaacgym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym.envs.go2.go2_nm_config import GO2NMCfg
from legged_gym.envs.go2.go2_nm_env import GO2NM
from legged_gym.utils import task_registry
from utils.cost_map import CostMap, TransitionCostMap, TransitionCostMapCfg
from utils.locomotion_model import LocoMotionModel
from utils.navigation import AStarPlanner, PlannerConfig, PlannerState
from utils.neural_map import MapCfg, TrustNeuralMap
from utils.position_controller import (
    PositionControllerConfig,
    compute_turn_first_position_command,
    sample_target_xy,
)

import numpy as np

from utils.cost_predict_nn import CostPredNNSingleLatent

# Map configuration constants
MAP_X_RANGE = (-2.0, 10.0)
MAP_Y_RANGE = (-2.0, 10.0)
MAP_LEAST_COUNT = 0.5
# MAP_COST_INIT = 300.0
# OBSTACLE_THRESHOLD = 500.0
MAP_COST_INIT = 45.
OBSTACLE_THRESHOLD = 70.
# Subsample the A* path: take every Nth waypoint to avoid overly dense targets
# WAYPOINT_STRIDE = 3
WAYPOINT_STRIDE = 1
# Replan every N simulation steps on the updated cost map (rolling update)
REPLAN_INTERVAL = 200


critic_hidden_dims = [512, 256, 128]
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

    def load_weights(self):
        state_dict = torch.load("/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/notebooks/critic_weights.pt")
        self.load_state_dict(state_dict)

    def forward(self, obs):
        return self.critic(obs)


class DatasetLogger():
    def __init__(self, enable_flag = False,save_dir="/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/processed_cost_dataset"):
        self.save_dir = save_dir
        self.long_latents = []
        self.short_latents = []
        self.costs = []
        self.enable_flag = enable_flag
        pass

    def update(self, long_latent, short_latent, cost):
        if self.enable_flag:
            if len(long_latent.shape) == 2:
                long_latent = long_latent[0]
                short_latent = short_latent[0]
            self.long_latents.append(long_latent)
            self.short_latents.append(short_latent)
            self.costs.append(cost)
        else:
            pass

    def save_dataset(self):
        if self.enable_flag:
            from datetime import datetime
            import os
            filename = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.pt"
            filepath = os.path.join(self.save_dir, filename)
            os.makedirs(self.save_dir, exist_ok=True)

            self.long_latents = torch.stack(self.long_latents)
            self.short_latents = torch.stack(self.short_latents)

            input = torch.cat([self.long_latents, self.short_latents], dim=-1)
            label = torch.stack(self.costs)

            torch.save({
                "input": input,
                "label": label
            }, filepath)
            print(f"Saved dataset to {filepath}")
        else:
            print("Dataset logger module not enabled")



def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2nm", "help": "Task name."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint."},
        {"name": "--experiment_name", "type": str, "default": "locomotion", "help": "Experiment name."},
        {"name": "--run_name", "type": str, "default": "long_short_transformer", "help": "Run name."},
        {"name": "--load_run", "type": str, "help": "Run name to load when resume=True."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint number. If omitted, loads the latest checkpoint."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times."},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training."},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device."},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create."},
        {"name": "--seed", "type": int, "help": "Random seed."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations."},
        {"name": "--goal_tolerance", "type": float, "default": 0.5, "help": "Distance threshold for considering a waypoint reached."},
        {"name": "--position_gain", "type": float, "default": 0.8, "help": "Proportional gain from position error to reference velocity."},
        {"name": "--max_speed", "type": float, "default": 0.8, "help": "Maximum commanded planar speed."},
        {"name": "--max_lateral_speed", "type": float, "default": 0.5, "help": "Maximum commanded lateral speed in root frame."},
        {"name": "--heading_slowdown", "type": float, "default": 0.35, "help": "Minimum speed scale while turning away from the target."},
        {"name": "--turn_in_place_angle", "type": float, "default": 2.2, "help": "If |heading error| exceeds this angle in radians, rotate first."},
        {"name": "--lateral_zero_angle", "type": float, "default": 1.2, "help": "If |heading error| exceeds this angle, suppress lateral velocity."},
        {"name": "--terrain_yaml", "type": str,
         "default": "/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/terrains/configs/plane.yaml",
         "help": "Terrain yaml used for evaluation."},
        {"name": "--replan_interval", "type": int, "default": REPLAN_INTERVAL,
         "help": "Replan on the updated cost map every N simulation steps."},
        {"name": "--save_dataset", "action": "store_true", "default": False, "help": "Save cost predict for CostPredictNN training"},
    ]

    args = gymutil.parse_arguments(
        description="Run policy with planned path navigation (A* on cost map).",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def xy_to_grid(xy: torch.Tensor, x_range: Tuple[float, float], y_range: Tuple[float, float], least_count: float) -> Tuple[int, int]:
    """Convert world xy to cost-map grid (row=x_idx, col=y_idx)."""
    x_idx = int(torch.floor((xy[0] - x_range[0]) / least_count).item())
    y_idx = int(torch.floor((xy[1] - y_range[0]) / least_count).item())
    return (x_idx, y_idx)


def grid_to_xy(
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


def clamp_grid(grid: Tuple[int, int], h: int, w: int) -> Tuple[int, int]:
    return (max(0, min(grid[0], h - 1)), max(0, min(grid[1], w - 1)))


def plan_waypoints(
    cost_map: CostMap,
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
    cost_map_2d = cost_map.maps[0, :, :, 0].detach().cpu()
    h, w = cost_map_2d.shape

    start_grid = clamp_grid(xy_to_grid(start_xy.cpu(), x_range, y_range, least_count), h, w)
    goal_grid = clamp_grid(xy_to_grid(goal_xy.cpu(), x_range, y_range, least_count), h, w)

    cfg = PlannerConfig(obstacle_threshold=OBSTACLE_THRESHOLD)
    planner = AStarPlanner(cfg)
    result = planner.plan(start_grid, goal_grid, cost_map_2d)

    if result.state != PlannerState.SUCCESS or result.path is None:
        print(f"[Planner] Failed: {result.state.name} | start_grid={start_grid} goal_grid={goal_grid}")
        return None

    path = result.path
    print(f"[Planner] Success: {len(path)} steps, cost={result.total_cost:.2f}")

    # Subsample path waypoints (always include last)
    indices = list(range(0, len(path), WAYPOINT_STRIDE))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)

    waypoints = [
        grid_to_xy(path[i], x_range, y_range, least_count, device)
        for i in indices
    ]
    return waypoints


def draw_markers(env: GO2NM, current_xy: torch.Tensor, waypoint_xy: torch.Tensor, goal_xy: torch.Tensor):
    terrain_id = env.env_terrain_map[0]
    terrain_origin = env.terrain_loader._instances[terrain_id].array_offset

    robot_marker = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
    waypoint_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 0))
    goal_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 1))

    robot_h = env.terrain_loader.get_height(terrain_id, current_xy)
    waypoint_h = env.terrain_loader.get_height(terrain_id, waypoint_xy)
    goal_h = env.terrain_loader.get_height(terrain_id, goal_xy)

    robot_pose = gymapi.Transform(
        gymapi.Vec3(current_xy[0].item() + terrain_origin[0], current_xy[1].item() + terrain_origin[1], robot_h), r=None
    )
    waypoint_pose = gymapi.Transform(
        gymapi.Vec3(waypoint_xy[0].item() + terrain_origin[0], waypoint_xy[1].item() + terrain_origin[1], waypoint_h), r=None
    )
    goal_pose = gymapi.Transform(
        gymapi.Vec3(goal_xy[0].item() + terrain_origin[0], goal_xy[1].item() + terrain_origin[1], goal_h), r=None
    )

    env.gym.clear_lines(env.viewer)
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 0., 10.), color=gymapi.Vec3(0., 0., 1.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 10., 0.), color=gymapi.Vec3(0., 1., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(10., 0., 0.), color=gymapi.Vec3(1., 0., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_lines(robot_marker, env.gym, env.viewer, env.envs[0], robot_pose)
    gymutil.draw_lines(waypoint_marker, env.gym, env.viewer, env.envs[0], waypoint_pose)
    gymutil.draw_lines(goal_marker, env.gym, env.viewer, env.envs[0], goal_pose)


def play(args):
    env_cfg: GO2NMCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrains_yaml = args.terrain_yaml
    env_cfg.cost.scales.power = 0.0
    env_cfg.cost.scales.collision = 0.7

    env_cfg.init_state.rot = [0., 0., 1., 0.]

    env: GO2NM
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()

    train_cfg.runner.resume = True

    neural_map_cfg = MapCfg()
    neural_map_cfg.x_range = MAP_X_RANGE
    neural_map_cfg.y_range = MAP_Y_RANGE
    neural_map_cfg.device = env.device
    neural_map_cfg.short_latent_dim = 32
    neural_map_cfg.long_latent_dim = 32
    neural_map_cfg.least_count = MAP_LEAST_COUNT
    neural_map = TrustNeuralMap(num_envs=env.num_envs, cfg=neural_map_cfg)
    cost_map_cfg = TransitionCostMapCfg(
        x_range=MAP_X_RANGE,
        y_range=MAP_Y_RANGE,
        device=env.device,
        least_count=MAP_LEAST_COUNT,
        init_value=MAP_COST_INIT,
    )
    cost_map = TransitionCostMap(num_envs=env.num_envs, cfg=cost_map_cfg)

    locomotion_model = LocoMotionModel(device=env.device)
    locomotion_model.to(env.device)

    controller_cfg = PositionControllerConfig(
        goal_tolerance=args.goal_tolerance,
        position_gain=args.position_gain,
        max_speed=args.max_speed,
        max_lateral_speed=args.max_lateral_speed,
        heading_slowdown=args.heading_slowdown,
        turn_in_place_angle=args.turn_in_place_angle,
        lateral_zero_angle=args.lateral_zero_angle,
    )

    # Matplotlib cost-map visualization
    plt.ion()
    fig, ax = plt.subplots()
    cost_map_vals = cost_map.maps[0, :, :, 0].detach().cpu()
    im = ax.imshow(cost_map_vals, cmap="hot")
    # im.set_clim(vmin=0.0, vmax=500.0)
    im.set_clim(vmin=40.0, vmax=75.0)
    plt.colorbar(im)
    re_plot_cnt = 0

    sensor_force = []

    # Init Critic
    critic = Critic(critic_hidden_dims)
    critic.load_weights()
    critic.to(env.device)

    dataset_logger = DatasetLogger(enable_flag=args.save_dataset)

    cost_pred_nn = CostPredNNSingleLatent.builder() \
        .ckpt_path("/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/logs/cost_pred-2026-04-07_11-04-53/model_4500.pt") \
        .build()
    cost_pred_nn.load_weights()
    cost_pred_nn.to(env.device)

    # Sample initial start and goal on the true map
    current_xy = env.root_local_pos[0, :2].clone()
    start_xy = sample_target_xy(
        device=env.device,
        x_min=MAP_X_RANGE[0] + 1.0,
        x_max=MAP_X_RANGE[1] - 1.0,
        y_min=MAP_Y_RANGE[0] + 1.0,
        y_max=MAP_Y_RANGE[1] - 1.0,
    )
    goal_xy = sample_target_xy(
        device=env.device,
        x_min=MAP_X_RANGE[0] + 2.0,
        x_max=MAP_X_RANGE[1] - 2.0,
        y_min=MAP_Y_RANGE[0] + 2.0,
        y_max=MAP_Y_RANGE[1] - 2.0,
        current_xy=start_xy,
        min_distance=2.0,
    )
    print(f"[Init] start={start_xy.tolist()} goal={goal_xy.tolist()}")

    # Plan initial path (cost map is uniform at init, so A* will route directly)
    waypoints: Optional[List[torch.Tensor]] = plan_waypoints(
        cost_map, start_xy, goal_xy,
        MAP_X_RANGE, MAP_Y_RANGE, MAP_LEAST_COUNT, env.device,
    )
    waypoint_idx = 0

    step_count = 0

    with torch.no_grad():
        while True:
            current_xy = env.root_local_pos[0, :2]

            # --- Rolling replan on interval ---
            if step_count > 0 and step_count % args.replan_interval == 0:
                print(f"[Replan] Step {step_count}: replanning on updated cost map.")
                new_waypoints = plan_waypoints(
                    cost_map, current_xy, goal_xy,
                    MAP_X_RANGE, MAP_Y_RANGE, MAP_LEAST_COUNT, env.device,
                )
                if new_waypoints is not None:
                    waypoints = new_waypoints
                    waypoint_idx = 0
                    print(f"[Replan] New plan: {len(waypoints)} waypoints toward goal={goal_xy.tolist()}")
                else:
                    print("[Replan] Planning failed, keeping previous waypoints.")

            # --- Advance waypoints ---
            if waypoints is not None and waypoint_idx < len(waypoints):
                target_xy = waypoints[waypoint_idx]
                dist_to_waypoint = torch.norm(current_xy - target_xy)
                if dist_to_waypoint <= args.goal_tolerance:
                    waypoint_idx += 1
                    if waypoint_idx < len(waypoints):
                        print(f"[Nav] Reached waypoint {waypoint_idx - 1}, next={waypoints[waypoint_idx].tolist()}")
            else:
                # Final goal reached (or no waypoints) — resample and replan
                print("[Nav] Goal reached! Resampling new goal and replanning.")
                start_xy = current_xy.clone()
                goal_xy = env.sample_goals(0)[0, :2]
                # goal_xy = sample_target_xy(
                #     device=env.device,
                #     x_min=MAP_X_RANGE[0] + 2.0,
                #     x_max=MAP_X_RANGE[1] - 2.0,
                #     y_min=MAP_Y_RANGE[0] + 2.0,
                #     y_max=MAP_Y_RANGE[1] - 2.0,
                #     current_xy=start_xy,
                #     min_distance=2.0,
                # )
                print(f"[Nav] New goal={goal_xy.tolist()}")
                waypoints = plan_waypoints(
                    cost_map, start_xy, goal_xy,
                    MAP_X_RANGE, MAP_Y_RANGE, MAP_LEAST_COUNT, env.device,
                )
                waypoint_idx = 0
                env.reset()

            # Determine active target (current waypoint or final goal as fallback)
            if waypoints is not None and waypoint_idx < len(waypoints):
                target_xy = waypoints[waypoint_idx]
            else:
                target_xy = goal_xy

            # --- Position controller ---
            control_output = compute_turn_first_position_command(
                current_xy=current_xy,
                root_quat_xyzw=env.root_states[0, 3:7],
                target_xy=target_xy,
                cfg=controller_cfg,
            )
            cmd_xy = control_output.command_xy
            heading_target = control_output.heading_target
            distance = control_output.distance

            env.commands[0, :2] = cmd_xy
            env.commands[0, 3] = heading_target

            # --- Locomotion step ---
            locomotion_model.compute_latents(obs.detach())
            actions = locomotion_model.act()
            obs, priviliged_obs, rews, dones, infos, _, _ = env.step(actions.detach())

            # --- Visualization markers ---
            draw_markers(env, current_xy, target_xy, goal_xy)

            # --- Neural map update ---
            local_pos = env.root_local_pos
            long_latent = locomotion_model.out["latent_long"]
            short_latent = locomotion_model.out["short_proj"]
            # cost = critic.forward(priviliged_obs).squeeze(0)
            # cost = 100 / cost
            cost = cost_pred_nn.inference(torch.cat([long_latent, short_latent], dim=-1)).squeeze(0)
            breakpoint()

            dataset_logger.update(long_latent.clone(), short_latent.clone(), cost.clone())

            sensor_force.append(env.sensor_forces.clone())

            print(
                f"pos=({current_xy[0].item():.3f}, {current_xy[1].item():.3f}) "
                f"wpt=({target_xy[0].item():.3f}, {target_xy[1].item():.3f}) "
                f"goal=({goal_xy[0].item():.3f}, {goal_xy[1].item():.3f}) "
                f"cmd=({cmd_xy[0].item():.3f}, {cmd_xy[1].item():.3f}) "
                f"dist={distance.item():.3f} wpt_idx={waypoint_idx}/{len(waypoints) if waypoints else 0} "
                f"step_cost={cost}"
            )

            step_count += 1

            try:
                neural_map.update(long_latents=long_latent.clone(), short_latents=short_latent.clone(), xy_coord=local_pos[:, :2])
                cost_map.update(current_pos=local_pos[:, :2], new_values=cost.clone())
            except Exception:
                print("update error")
                raise

            # --- Cost map display ---
            cost_map_vals = cost_map.maps[0, :, :, 0].detach().cpu()
            im.set_data(cost_map_vals)

            re_plot_cnt += 1
            if re_plot_cnt % 5 == 0:
                fig.canvas.draw()
                fig.canvas.flush_events()
                re_plot_cnt = 0

            def cbk():
                now = datetime.now()
                path = f"./temp/{now.strftime('%Y-%m-%d')}/"
                os.makedirs(path, exist_ok=True)
                neural_map.save(path)
                cost_map.save(path)
                torch.save(torch.stack(sensor_force), path + "sensor_forces.pt")
                # breakpoint()

            neural_map.reset(dones.clone(), reset_cbk=cbk)
            cost_map.reset(dones.clone())
            if step_count > 50 * 60 * 2:
                dataset_logger.save_dataset()
                break


if __name__ == "__main__":
    args = get_args()
    play(args)
