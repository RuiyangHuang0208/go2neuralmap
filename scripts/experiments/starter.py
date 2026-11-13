import os
from datetime import datetime

import isaacgym
import torch
from isaacgym import gymapi
from isaacgym import gymutil
import traceback

from legged_gym.envs.go2.go2_nm_config import GO2NMCfg
from legged_gym.envs.go2.go2_nm_env import GO2NM
from legged_gym.utils import task_registry

import hydra
from omegaconf import OmegaConf
from builder import (
    merge_env_config,
    build_logger,
    build_cost_map,
    build_neural_map,
    build_locomotion_model,
    build_cost_estimator,
    build_position_controller,
    build_path_planner,
    build_vel_cmd_generator,
    build_state_recorder,
    build_runner_dataset_collector,
)
from loguru import logger
from modules.vel_command_generator.vel_command_generator import normalize_vel_cmd_shape


def get_args():
    custom_parameters = [
        {"name": "--my_exp_name", "type": str, "default": "go2nm", "help": "exp name."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times."},
        {"name": "--deactivate_state_recorder", "action": "store_true", "default": False, "help": "deactivate state recorder"},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create."},
        {"name": "--log_level", "type": str, "help": "Log level of ste output"},
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


def load_configs(args):
    env_vars = os.environ
    commit_hash = env_vars.get("COMMIT_HASH")
    if env_vars is None:
        raise ValueError("No commit hash found")
    time_now = datetime.now()
    curr_log_dir_name = os.path.join(
        f"{time_now.strftime('%Y-%m-%d')}_{commit_hash}",
        f"{args.my_exp_name}_{time_now.strftime('%H-%M-%S')}"
    )

    hydra.initialize(config_path="configs", version_base="1.1")
    cfg_omega = hydra.compose(
        config_name="base_config",
        overrides=[
            f"+exp={args.my_exp_name}",
            f"helpers.state_recorder.name={'null_recorder' if args.deactivate_state_recorder else 'state_recorder'}",
            f"+curr_log_dir_name={curr_log_dir_name}"
        ]
    )
    OmegaConf.resolve(cfg_omega)
    yaml_content = OmegaConf.to_yaml(cfg_omega, resolve=False)
    print(yaml_content)
    return cfg_omega


class Runner():
    def __init__(self, cfg_omega: OmegaConf, args):
        self.args_cli = args
        self.cfg_omega = cfg_omega
        self.task = cfg_omega.task
        self.env_cfg_omega = cfg_omega.env
        self.logger = logger.bind(module_name="runner")

        self._init_env()
        self.logger.info("env initialized")

        self._init_modules()
        
        log_dir = self.cfg_omega.log_dir
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml_content = OmegaConf.to_yaml(self.cfg_omega, sort_keys=False)
            f.write(yaml_content)

    def _init_env(self):
        env_cfg: GO2NMCfg
        env_cfg, _ = task_registry.get_cfgs(name=self.task)
        env_cfg = merge_env_config(env_cfg, self.env_cfg_omega)
        self.env, _ = task_registry.make_env(name=self.task, args=self.args_cli, env_cfg=env_cfg)
        pass

    def _init_modules(self):
        self.cfg_omega.env.env.num_envs = self.env.num_envs
        self.cfg_omega.modules.neural_map.num_envs = self.env.num_envs
        self.cfg_omega.modules.cost_map.num_envs = self.env.num_envs

        nm_cfg_omega = self.cfg_omega.modules.neural_map
        self.neural_map = build_neural_map(nm_cfg_omega)
        self.logger.info("neural_map initialized")

        cm_cfg_omega = self.cfg_omega.modules.cost_map
        self.cost_map = build_cost_map(cm_cfg_omega)
        self.logger.info("cost_map initialized")

        locomotion_model_cfg_omega = self.cfg_omega.modules.locomotion_model
        self.locomotion_model = build_locomotion_model(locomotion_model_cfg_omega)
        self.locomotion_model.to(self.env.device)
        self.logger.info("locomotion_model initialized")

        cost_estimator_cfg_omega = self.cfg_omega.modules.cost_estimator
        self.cost_estimator = build_cost_estimator(cost_estimator_cfg_omega, self.env)
        self.logger.info("cost estimator initialized")

        position_controller_cfg_omega = self.cfg_omega.modules.position_controller
        self.position_controller = build_position_controller(position_controller_cfg_omega)
        self.logger.info("position controller initialized")

        planner_cfg_omega = self.cfg_omega.modules.planner
        self.planner = build_path_planner(planner_cfg_omega)
        self.logger.info("path planner initialized")

        vel_cmd_generator_cfg_omega = self.cfg_omega.modules.vel_command_generator
        self.vel_cmd_generator = build_vel_cmd_generator(vel_cmd_generator_cfg_omega, self)
        self.logger.info("vel cmd gen initialized")

        state_recorder_cfg_omega = self.cfg_omega.helpers.state_recorder
        self.state_recorder = build_state_recorder(state_recorder_cfg_omega)

        dataset_collector_cfg_omega = self.cfg_omega.helpers.dataset_collector
        self.dataset_collector = build_runner_dataset_collector(dataset_collector_cfg_omega)
        self.dataset_collector.validate(self)
        
        # breakpoint()
        pass

    def _retrieve_markers(self):
        markers = []
        markers.extend(self.vel_cmd_generator.retrieve_markers())
        return markers

    def _draw_markers(self):
        self.env.gym.clear_lines(self.env.viewer)
        gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 0., 10.), color=gymapi.Vec3(0., 0., 1.), gym=self.env.gym, viewer=self.env.viewer, env=self.env.envs[0])
        gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(0., 10., 0.), color=gymapi.Vec3(0., 1., 0.), gym=self.env.gym, viewer=self.env.viewer, env=self.env.envs[0])
        gymutil.draw_line(p1=gymapi.Vec3(0., 0., 0.), p2=gymapi.Vec3(10., 0., 0.), color=gymapi.Vec3(1., 0., 0.), gym=self.env.gym, viewer=self.env.viewer, env=self.env.envs[0])
        markers = self._retrieve_markers()
        for style, pose in markers:
            gymutil.draw_lines(style, self.env.gym, self.env.viewer, self.env.envs[0], pose)
        if hasattr(self.env, "draw_collision_contact_markers"):
            self.env.draw_collision_contact_markers()

    def _save_final_snapshot(self):
        snapshot_dir = os.path.join(self.cfg_omega.log_dir, "final_snapshot")
        os.makedirs(snapshot_dir, exist_ok=True)

        self.neural_map.save(os.path.join(snapshot_dir, "neural_map"))
        self.cost_map.save(os.path.join(snapshot_dir, "cost_map"))
        self.logger.info(f"Final NeuralMap and CostMap snapshot saved to {snapshot_dir}")

    def run(self):
        # Init state
        obs = self.env.get_observations()

        self.vel_cmd_generator.init_state()

        step_cnt = 0

        try:
            with torch.no_grad():
                while True:
                    vel_cmd = normalize_vel_cmd_shape(
                        self.vel_cmd_generator.compute_vel_cmd(),
                        self.env.commands.shape,
                    )
                    consume_reset = getattr(
                        self.vel_cmd_generator, "consume_robot_reset_request", None
                    )
                    if consume_reset is not None and consume_reset():
                        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
                        self.env.robot_reset(env_ids)
                        self.env.compute_observations()
                        obs = self.env.get_observations()
                        self.logger.info("Manual robot-state reset requested from keyboard")
                        continue
                    self.env.commands[:, :] = vel_cmd

                    # Step
                    self.locomotion_model.compute_latents(obs.detach())
                    actions = self.locomotion_model.act()
                    obs, privileged_obs, rews, dones, infos, _, _ = self.env.step(actions.detach())

                    # Post Processing
                    local_pos    = self.env.root_local_pos
                    long_latent  = self.locomotion_model.out["latent_long"]
                    short_latent = self.locomotion_model.out["short_proj"]

                    self.neural_map.update(
                        long_latents=long_latent.clone(),
                        short_latents=short_latent.clone(),
                        xy_coord=local_pos[:, :2],
                        root_velocity=self.env.root_states[:, 7:9],
                    )

                    cost: torch.Tensor = self.cost_estimator.compute_cost(
                        latent=torch.cat([long_latent, short_latent], dim=-1),
                        privileged_obs=privileged_obs,
                        neural_map=self.neural_map,
                    ) # Cost is of shape (num_envs,)
                    self.cost_map.update(
                        current_pos=local_pos[:, :2],
                        new_values=cost.clone(),
                        root_velocity=self.env.root_states[:, 7:9],
                        command=self.env.commands[:, :2],
                        directional_collision=self.env.get_directional_collision_flags(),
                        base_quat=self.env.base_quat,
                        latent=torch.cat([long_latent, short_latent], dim=-1),
                    )
                    if cost.numel() == 1:
                        self.logger.info(f"Cost: {cost.item()}")
                    else:
                        self.logger.info(
                            "Cost: mean={:.4f}, min={:.4f}, max={:.4f}",
                            cost.mean().item(),
                            cost.min().item(),
                            cost.max().item(),
                        )
                    self.dataset_collector.collect(
                        step=step_cnt,
                        neural_map=self.neural_map,
                        cost_map=self.cost_map,
                        env=self.env,
                        cost=cost,
                        dones=dones,
                        vel_command_generator_info=self.vel_cmd_generator.get_info(),
                    )
                    # self.logger.debug(f"Cost: {cost.item()}")

                    # Draw markers
                    self._draw_markers()
                    uncertainty_map_vals = None
                    if hasattr(self.neural_map, "uncertainties"):
                        uncertainty_source = (
                            self.neural_map.get_aged_uncertainties()
                            if hasattr(self.neural_map, "get_aged_uncertainties")
                            else self.neural_map.uncertainties
                        )
                        if uncertainty_source.dim() == 4 and uncertainty_source.shape[-1] == 4:
                            uncertainty_map_vals = uncertainty_source[0].permute(2, 0, 1).detach().cpu()

                    self.state_recorder.collect_vars({
                        "step": step_cnt,
                        "current_xy": self.env.root_local_pos[0, :2].detach().cpu(),
                        "cost_map_vals": self.cost_map.maps[0, :, :].detach().cpu(),
                        "vel_cmd_generator_info": self.vel_cmd_generator.get_info(),
                        "reset_buf": dones.clone().detach().cpu(),
                        "collision_flags": self.env.get_directional_collision_flags().clone().detach().cpu(),
                        "dt": float(self.env.dt),
                        "current_cost": cost.clone().detach().cpu(),
                        "long_latent": long_latent.clone().detach().cpu(),
                        "short_latent": short_latent.clone().detach().cpu(),
                        "uncertainty_map_vals": uncertainty_map_vals,
                        "uncertainty_map_metadata": {
                            "x_range": tuple(float(value) for value in self.neural_map.cfg.x_range),
                            "y_range": tuple(float(value) for value in self.neural_map.cfg.y_range),
                            "least_count": float(self.neural_map.cfg.least_count),
                        },
                    })
                    self.state_recorder.handle()

                    self.cost_map.reset(dones.clone(), reset_values=False)
                    # self.neural_map.reset(torch.ones_like(dones, dtype=torch.int))
                    step_cnt += 1
                    if step_cnt % 10000 == 0:
                        self.state_recorder.save()
                        step_cnt = 0
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            self.logger.info("Simulation terminated")
            try:
                self._save_final_snapshot()
            except Exception as e:
                self.logger.exception(f"Failed to save final snapshot: {e}")
            finally:
                self.dataset_collector.flush()
                self.state_recorder.save()
            pass


def main():
    args = get_args()
    # my_exp_name = args.my_exp_name
    cfg_omega = load_configs(args)

    logger_cfg = cfg_omega.helpers.Logger
    logger = build_logger(logger_cfg, args.log_level)
    main_proc_logger = logger.bind(module_name="main_proc")

    main_proc_logger.info("Configuration Parsed")

    runner = Runner(cfg_omega, args)

    runner.run()


if __name__ == "__main__":
    main()
