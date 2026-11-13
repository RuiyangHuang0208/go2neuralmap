import isaacgym
from isaacgym import gymutil
from isaacgym import gymapi
from legged_gym.envs.go2.go2_nm_env import GO2NM
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from utils.cost_map import TransitionCostMap, TransitionCostMapCfg
from utils.locomotion_model import LocoMotionModel
from utils.neural_map import MapCfg, TrustNeuralMap
import torch
import os
from datetime import datetime
import pygame
import numpy as np
from legged_gym.envs.go2.go2_nm_config import GO2NMCfg
from utils.user_input import LinuxKeyboardController
from utils.cost_predict_nn import CostPredNNSingleLatent, Config as CostPredNNConfig

import matplotlib.pyplot as plt


def cmd_sampler_builder():
    kbd_controller = LinuxKeyboardController()
    heading = 0
    last_heading_state = 0

    def foo():

        cmd = kbd_controller.get_command()

        nonlocal heading
        nonlocal last_heading_state

        x_val = cmd[0]
        y_val = cmd[1]
        rot = cmd[2]

        if rot > 0 and rot != last_heading_state:
            heading += 0.1 * np.pi
        elif rot < 0 and rot != last_heading_state:
            heading -= 0.1 * np.pi

        last_heading_state = rot

        # heading = np.clip(heading, -np.pi, np.pi)
        if heading < -np.pi:
            heading += 2 * np.pi
        if heading > np.pi:
            heading -= 2 * np.pi

        return [x_val, y_val, heading]

    return foo


def play(args):

    env_cfg: GO2NMCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # train_cfg.policy.policy_mode = "current+short+long"
    # train_cfg.policy.long_encoder_type = "transformer"
    # train_cfg.policy.long_encoder = "transformer"

    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrains_yaml = "/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/terrains/configs/sample.yaml"
    env_cfg.cost.scales.power = 1.0
    env_cfg.cost.scales.collision = 0.7

    env: GO2NM
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    policy_cfg = train_cfg.policy
    policy_cfg = class_to_dict(policy_cfg)
    # ppo_runner, train_cfg = task_registry.make_alg_runner_dual(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

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

    sample_cmd = cmd_sampler_builder()

    sensor_force = []
    print(f"key body names: {env.key_body_names}")

    # Cost Pred NN
    cost_pred_nn_cfg = CostPredNNConfig()
    cost_pred_nn = CostPredNNSingleLatent(cfg=cost_pred_nn_cfg)
    cost_pred_nn_state_dict = torch.load("/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/logs/cost_pred-2026-03-07_17-23-20/model_19450.pt")
    cost_pred_nn.load_state_dict(cost_pred_nn_state_dict["model"])
    cost_pred_nn.to(device=env.device)

    # Draw map
    plt.ion()
    fig, ax = plt.subplots()
    cost_map_vals = cost_map.maps[0].detach().cpu()
    im = ax.imshow(cost_map_vals, cmap="hot")
    im.set_clim(vmin=0.0, vmax=500.0)
    plt.colorbar(im)
    re_plot_cnt = 0

    with torch.no_grad():
        while True:

            cmd = torch.tensor(sample_cmd(), dtype=torch.float, device=env.device)
            # env.commands[0, :3] = cmd[:3]
            env.commands[0, :2] = cmd[:2]
            env.commands[0, 3] = cmd[2]
            # env.commands[0, 2] = 3.0
            # print(f"xvel: {cmd[0]}, yvel: {cmd[1]}, heading: {cmd[2]}")

            locomotion_model.compute_latents(obs.detach())
            actions = locomotion_model.act()
            # env.commands = torch.zeros((env.num_envs, 4), dtype=torch.float, device=env.device)
            # env.commands[:, 0] = 0.6
            obs, priviliged_obs, rews, dones, infos, _, _ = env.step(actions.detach())
            terrain_id = env.env_terrain_map[0] # terrain id for env 0
            sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
            height = env.terrain_loader.get_height(terrain_id, torch.tensor([8.0, 0.0], device=env.device))
            terrain_origin = env.terrain_loader._instances[terrain_id].array_offset
            sphere_pose = gymapi.Transform(gymapi.Vec3(8.0 + terrain_origin[0], 0.0 + terrain_origin[1], height), r=None)

            # Draw instructors
            env.gym.clear_lines(env.viewer)
            gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 0., 10.), color=gymapi.Vec3(0., 0., 1.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
            gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 10., 0.), color=gymapi.Vec3(0., 1., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
            gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(10., 0., 0.), color=gymapi.Vec3(1., 0., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
            gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[0], sphere_pose)

            local_pos = env.root_local_pos
            long_latent = locomotion_model.out["latent_long"]
            short_latent = locomotion_model.out["short_proj"]

            # Pred Cost
            cost = cost_pred_nn.inference(torch.cat([long_latent, short_latent], dim=-1))
            sensor_force.append(env.sensor_forces.clone())
            print(f"step_cost: {cost}")
            try:
                neural_map.update(long_latents=long_latent.clone(), short_latents=short_latent.clone(), xy_coord=local_pos[:, :2])
                cost_map.update(current_pos=local_pos[:, :2], new_values=cost[0].clone())

            except:
                print("update error")
                exit()

            cost_map_vals = cost_map.maps[0].detach().cpu()
            im.set_data(cost_map_vals)
            # im.set_clim(vmin=)

            re_plot_cnt += 1
            if re_plot_cnt % 5 == 0:
                fig.canvas.draw()
                fig.canvas.flush_events()
                re_plot_cnt = 0

            def cbk():
                # now = datetime.now()
                # path = f"./temp/{now.strftime('%Y-%m-%d')}/"
                # os.makedirs(path, exist_ok=True)
                # neural_map.save(path)
                # cost_map.save(path)
                # torch.save(torch.stack(sensor_force), path + "sensor_forces.pt")
                breakpoint()
                pass
            neural_map.reset(dones.clone(), reset_cbk=cbk)
            cost_map.reset(dones.clone())
            pass
    pass


if __name__ == "__main__":
    args = get_args()
    args.task = "go2nm"
    args.experiment_name = "locomotion"
    args.run_name = "long_short_transformer"
    play(args)
