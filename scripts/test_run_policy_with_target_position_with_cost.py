import os
from datetime import datetime

import isaacgym
import matplotlib.pyplot as plt
import torch
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym.envs.go2.go2_nm_config import GO2NMCfg
from legged_gym.envs.go2.go2_nm_env import GO2NM
from legged_gym.utils import task_registry
from utils.cost_map import TransitionCostMap, TransitionCostMapCfg
from utils.locomotion_model import LocoMotionModel
from utils.neural_map import MapCfg, TrustNeuralMap
from utils.position_controller import (
    PositionControllerConfig,
    compute_turn_first_position_command,
    sample_target_xy,
)


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
        {"name": "--target_x", "type": float, "default": 8.0, "help": "Target x position in terrain-local coordinates."},
        {"name": "--target_y", "type": float, "default": 0.0, "help": "Target y position in terrain-local coordinates."},
        {"name": "--goal_tolerance", "type": float, "default": 0.25, "help": "Distance threshold for considering the goal reached."},
        {"name": "--position_gain", "type": float, "default": 0.8, "help": "Proportional gain from position error to reference velocity."},
        {"name": "--max_speed", "type": float, "default": 0.8, "help": "Maximum commanded planar speed."},
        {"name": "--max_lateral_speed", "type": float, "default": 0.5, "help": "Maximum commanded lateral speed in root frame."},
        {"name": "--heading_slowdown", "type": float, "default": 0.35, "help": "Minimum speed scale while turning away from the target."},
        {"name": "--turn_in_place_angle", "type": float, "default": 2.2, "help": "If |heading error| exceeds this angle in radians, rotate first and command no planar velocity."},
        {"name": "--lateral_zero_angle", "type": float, "default": 1.2, "help": "If |heading error| exceeds this angle in radians, suppress lateral velocity."},
        {"name": "--resample_goal_on_reach", "action": "store_true", "default": False, "help": "Randomly resample a new goal after reaching the current goal."},
        {"name": "--goal_x_min", "type": float, "default": -2.0, "help": "Minimum x value for random goal resampling."},
        {"name": "--goal_x_max", "type": float, "default": 10.0, "help": "Maximum x value for random goal resampling."},
        {"name": "--goal_y_min", "type": float, "default": -2.0, "help": "Minimum y value for random goal resampling."},
        {"name": "--goal_y_max", "type": float, "default": 10.0, "help": "Maximum y value for random goal resampling."},
        {"name": "--resample_min_distance", "type": float, "default": 1.0, "help": "Minimum distance from the current position for a resampled goal."},
        {"name": "--terrain_yaml", "type": str, "default": "/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/terrains/configs/position_nav.yaml", "help": "Terrain yaml used for evaluation."},
    ]

    args = gymutil.parse_arguments(description="Run policy toward a target 2D position.", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args

def play(args):
    env_cfg: GO2NMCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrains_yaml = args.terrain_yaml
    env_cfg.cost.scales.power = 1.0
    env_cfg.cost.scales.collision = 0.7

    env: GO2NM
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()

    train_cfg.runner.resume = True

    neural_map_cfg = MapCfg()
    neural_map_cfg.x_range = (-2.0, 10.0)
    neural_map_cfg.y_range = (-2.0, 10.0)
    neural_map_cfg.device = env.device
    neural_map_cfg.short_latent_dim = 32
    neural_map_cfg.long_latent_dim = 32
    neural_map_cfg.least_count = 0.5
    neural_map = TrustNeuralMap(num_envs=env.num_envs, cfg=neural_map_cfg)
    cost_map_cfg = TransitionCostMapCfg(
        x_range=(-2.0, 10.0),
        y_range=(-2.0, 10.0),
        device=env.device,
        least_count=0.5,
        init_value=300.0,
    )
    cost_map = TransitionCostMap(num_envs=env.num_envs, cfg=cost_map_cfg)

    locomotion_model = LocoMotionModel(device=env.device)
    locomotion_model.to(env.device)

    target_xy = torch.tensor([args.target_x, args.target_y], dtype=torch.float, device=env.device)
    controller_cfg = PositionControllerConfig(
        goal_tolerance=args.goal_tolerance,
        position_gain=args.position_gain,
        max_speed=args.max_speed,
        max_lateral_speed=args.max_lateral_speed,
        heading_slowdown=args.heading_slowdown,
        turn_in_place_angle=args.turn_in_place_angle,
        lateral_zero_angle=args.lateral_zero_angle,
    )

    sensor_force = []
    print(f"key body names: {env.key_body_names}")
    print(f"target_xy: {target_xy.tolist()}")

    plt.ion()
    fig, ax = plt.subplots()
    cost_map_vals = cost_map.maps[0].detach().cpu()
    im = ax.imshow(cost_map_vals, cmap="hot")
    im.set_clim(vmin=0.0, vmax=500.0)
    plt.colorbar(im)
    re_plot_cnt = 0

    with torch.no_grad():
        while True:
            control_output = compute_turn_first_position_command(
                current_xy=env.root_local_pos[0, :2],
                root_quat_xyzw=env.root_states[0, 3:7],
                target_xy=target_xy,
                cfg=controller_cfg,
            )
            cmd_xy = control_output.command_xy
            heading_target = control_output.heading_target
            distance = control_output.distance
            current_xy = env.root_local_pos[0, :2]

            if distance <= args.goal_tolerance and args.resample_goal_on_reach:
                target_xy = sample_target_xy(
                    device=env.device,
                    x_min=args.goal_x_min,
                    x_max=args.goal_x_max,
                    y_min=args.goal_y_min,
                    y_max=args.goal_y_max,
                    current_xy=current_xy,
                    min_distance=max(args.resample_min_distance, args.goal_tolerance),
                )
                print(f"resampled target_xy: {target_xy.tolist()}")
                control_output = compute_turn_first_position_command(
                    current_xy=env.root_local_pos[0, :2],
                    root_quat_xyzw=env.root_states[0, 3:7],
                    target_xy=target_xy,
                    cfg=controller_cfg,
                )
                cmd_xy = control_output.command_xy
                heading_target = control_output.heading_target
                distance = control_output.distance
                current_xy = env.root_local_pos[0, :2]

            env.commands[0, :2] = cmd_xy
            env.commands[0, 3] = heading_target

            locomotion_model.compute_latents(obs.detach())
            actions = locomotion_model.act()
            obs, priviliged_obs, rews, dones, infos, _, _ = env.step(actions.detach())

            terrain_id = env.env_terrain_map[0]
            robot_marker = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
            target_marker = gymutil.WireframeSphereGeometry(0.15, 4, 4, None, color=(0, 1, 1))

            terrain_origin = env.terrain_loader._instances[terrain_id].array_offset
            robot_marker_height = env.terrain_loader.get_height(terrain_id, current_xy)
            target_marker_height = env.terrain_loader.get_height(terrain_id, target_xy)

            robot_marker_pose = gymapi.Transform(
                gymapi.Vec3(current_xy[0].item() + terrain_origin[0], current_xy[1].item() + terrain_origin[1], robot_marker_height),
                r=None,
            )
            target_marker_pose = gymapi.Transform(
                gymapi.Vec3(target_xy[0].item() + terrain_origin[0], target_xy[1].item() + terrain_origin[1], target_marker_height),
                r=None,
            )

            env.gym.clear_lines(env.viewer)
            gymutil.draw_line(p1=gymapi.Vec3(0.0, 0.0, 0.0), p2=gymapi.Vec3(0.0, 0.0, 10.0), color=gymapi.Vec3(0.0, 0.0, 1.0), gym=env.gym, viewer=env.viewer, env=env.envs[0])
            gymutil.draw_line(p1=gymapi.Vec3(0.0, 0.0, 0.0), p2=gymapi.Vec3(0.0, 10.0, 0.0), color=gymapi.Vec3(0.0, 1.0, 0.0), gym=env.gym, viewer=env.viewer, env=env.envs[0])
            gymutil.draw_line(p1=gymapi.Vec3(0.0, 0.0, 0.0), p2=gymapi.Vec3(10.0, 0.0, 0.0), color=gymapi.Vec3(1.0, 0.0, 0.0), gym=env.gym, viewer=env.viewer, env=env.envs[0])
            gymutil.draw_lines(robot_marker, env.gym, env.viewer, env.envs[0], robot_marker_pose)
            gymutil.draw_lines(target_marker, env.gym, env.viewer, env.envs[0], target_marker_pose)

            local_pos = env.root_local_pos
            long_latent = locomotion_model.out["latent_long"]
            short_latent = locomotion_model.out["short_proj"]
            cost = env.step_cost
            sensor_force.append(env.sensor_forces.clone())
            print(
                f"pos=({current_xy[0].item():.3f}, {current_xy[1].item():.3f}) "
                f"target=({target_xy[0].item():.3f}, {target_xy[1].item():.3f}) "
                f"cmd=({cmd_xy[0].item():.3f}, {cmd_xy[1].item():.3f}) "
                f"heading={heading_target.item():.3f} dist={distance.item():.3f} step_cost={cost}"
            )

            try:
                neural_map.update(long_latents=long_latent.clone(), short_latents=short_latent.clone(), xy_coord=local_pos[:, :2])
                cost_map.update(current_pos=local_pos[:, :2], new_values=cost.clone())
            except Exception:
                print("update error")
                raise

            cost_map_vals = cost_map.maps[0].detach().cpu()
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
                breakpoint()

            neural_map.reset(dones.clone(), reset_cbk=cbk)
            cost_map.reset(dones.clone())


if __name__ == "__main__":
    args = get_args()
    play(args)
