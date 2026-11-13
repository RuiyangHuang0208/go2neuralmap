import isaacgym
from isaacgym import gymutil
from isaacgym import gymapi
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules.actor_critic_long_short import ActorCritic
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg.policy.policy_mode = "current+short+long"
    train_cfg.policy.long_encoder_type = "transformer"
    train_cfg.policy.long_encoder = "transformer"

    env_cfg.env.num_envs = 2
    env_cfg.terrain.curriculum = True

    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    obs = env.get_observations()

    # Load policy
    train_cfg.runner.resume = True
    policy_cfg = train_cfg.policy
    policy_cfg = class_to_dict(policy_cfg)
    # ppo_runner, train_cfg = task_registry.make_alg_runner_dual(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    num_critic_obs = env.num_privileged_obs
    num_long_history_length = env.long_history_length
    num_short_history_length = env.short_history_length
    dim_long_history_latent = env.cfg.env.dim_long_history_latent

    actor_critic: ActorCritic = ActorCritic(env.dim_proprio_obs,
                                            num_critic_obs,
                                            env.num_actions,
                                            num_long_history_length,
                                            num_short_history_length,
                                            dim_long_history_latent,
                                            **policy_cfg).to(env.device)

    loaded_dict = torch.load("/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/locomotion_model/long_short_transformer/model_15000.pt")
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    policy = actor_critic.actor
    offset = torch.tensor([-1.0, 0.0, 1.5], dtype=torch.float, device=env.device)

    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 0., 10.), color=gymapi.Vec3(0., 0., 1.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 10., 0.), color=gymapi.Vec3(0., 1., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])
    gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(10., 0., 0.), color=gymapi.Vec3(1., 0., 0.), gym=env.gym, viewer=env.viewer, env=env.envs[0])

    with torch.no_grad():
        # for i in range(int(env.max_episode_length)):
        while True:
            actions: torch.Tensor = policy(obs.detach())
            env.commands = torch.zeros((env.num_envs, 4), dtype=torch.float, device=env.device)
            env.commands[:, 0] = 0.6
            # root_pos = env.root_states[0, :3]
            # env.set_camera(root_pos + offset, root_pos)
            obs, priviliged_obs, rews, dones, infos, _, _ = env.step(actions.detach())
            pass
    pass


if __name__ == "__main__":
    args = get_args()
    args.task = "go2rough"
    args.experiment_name = "locomotion"
    args.run_name = "long_short_transformer"
    play(args)
