import isaacgym
from isaacgym import gymutil
from isaacgym import gymapi
from legged_gym.envs.go2.go2_rough_env import GO2Rough
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from utils.locomotion_model import LocoMotionModel
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg.policy.policy_mode = "current+short+long"
    train_cfg.policy.long_encoder_type = "transformer"
    train_cfg.policy.long_encoder = "transformer"

    env_cfg.env.num_envs = 2
    env_cfg.terrain.curriculum = True

    env: GO2Rough
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    policy_cfg = train_cfg.policy
    policy_cfg = class_to_dict(policy_cfg)
    # ppo_runner, train_cfg = task_registry.make_alg_runner_dual(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    locomotion_model = LocoMotionModel(device=env.device)
    locomotion_model.to(env.device)

    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 0., 10.), color=gymapi.Vec3(0., 0., 1.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 10., 0.), color=gymapi.Vec3(0., 1., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(10., 0., 0.), color=gymapi.Vec3(1., 0., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])

    with torch.no_grad():
        while True:
            locomotion_model.compute_latents(obs.detach())
            actions = locomotion_model.act()
            env.commands = torch.zeros((env.num_envs, 4), dtype=torch.float, device=env.device)
            env.commands[:, 0] = 0.6
            obs, priviliged_obs, rews, dones, infos, _, _ = env.step(actions.detach())
            env.gym.clear_lines(env.viewer)
            terrain_id = env.env_terrain_map[0] # terrain id for env 0
            sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
            height = env.terrain_loader.get_height(terrain_id, torch.tensor([8.0, 0.0], device=env.device))
            terrain_origin = env.terrain_loader._instances[terrain_id].array_offset
            sphere_pose = gymapi.Transform(gymapi.Vec3(8.0 + terrain_origin[0], 0.0 + terrain_origin[1], height), r=None)
            gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[0], sphere_pose)
            pass
    pass


if __name__ == "__main__":
    args = get_args()
    args.task = "go2rough"
    args.experiment_name = "locomotion"
    args.run_name = "long_short_transformer"
    play(args)
