import isaacgym
from isaacgym import gymutil
from isaacgym import gymapi
from legged_gym.envs.go2.go2_rough_env import GO2Rough
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from utils.locomotion_model import LocoMotionModel
from utils.neural_map import MapCfg, TrustNeuralMap
import torch
import os
from datetime import datetime
import pygame
import numpy as np
from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg


def cmd_sampler_builder():
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joinstick initialized")

    heading = 0.0
    prev_button_4 = 0.0
    prev_button_5 = 0.0

    def foo():
        nonlocal heading
        nonlocal prev_button_4
        nonlocal prev_button_5
        nonlocal joystick

        pygame.event.pump()

        # Read axes states
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        # Read buttons states
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        y_val = -1 * axes[0]
        x_val = -1 * axes[1]

        ang_vel = 0.0

        # Jif prev_button_4 == 0 and buttons[4] == 1:
        # J    heading += 0.1 * np.pi
        # Jelif prev_button_5 == 0 and buttons[5] == 1:
        # J    heading -= 0.1 * np.pi
        if prev_button_4 == 0 and buttons[4] == 1:
            heading += 0.1 * np.pi
        elif prev_button_5 == 0 and buttons[5] == 1:
            heading -= 0.1 * np.pi

        prev_button_4 = buttons[4]
        prev_button_5 = buttons[5]

        # heading = np.clip(heading, -np.pi, np.pi)
        if heading < -np.pi:
            heading += 2 * np.pi
        if heading > np.pi:
            heading -= 2 * np.pi

        return [x_val, y_val, heading]

    return foo


def play(args):

    env_cfg: GO2RoughCfg
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # train_cfg.policy.policy_mode = "current+short+long"
    # train_cfg.policy.long_encoder_type = "transformer"
    # train_cfg.policy.long_encoder = "transformer"

    env_cfg.env.num_envs = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrains_yaml = "/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/terrains/sample.yaml"

    env: GO2Rough
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
    neural_map_cfg.least_count = 0.25
    neural_map = TrustNeuralMap(num_envs=env.num_envs, cfg=neural_map_cfg)

    locomotion_model = LocoMotionModel(device=env.device)
    locomotion_model.to(env.device)


    sample_cmd = cmd_sampler_builder()

    with torch.no_grad():
        while True:

            cmd = torch.tensor(sample_cmd(), dtype=torch.float, device=env.device)
            # env.commands[0, :3] = cmd[:3]
            env.commands[0, :2] = cmd[:2]
            env.commands[0, 3] = cmd[2]
            # env.commands[0, 2] = 3.0
            print(cmd)

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
            try:
                neural_map.update(long_latents=long_latent.clone(), short_latents=short_latent.clone(), xy_coord=local_pos[:, :2])

            except:
                print("update error")
                exit()

            def cbk():
                now = datetime.now()
                path = f"./temp/{now.strftime('%Y-%m-%d')}/"
                os.makedirs(path, exist_ok=True)
                neural_map.save(path)
                breakpoint()
                pass
            neural_map.reset(dones.clone(), reset_cbk=cbk)
            pass
    pass


if __name__ == "__main__":
    args = get_args()
    args.task = "go2rough"
    args.experiment_name = "locomotion"
    args.run_name = "long_short_transformer"
    play(args)
