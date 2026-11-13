import sys

from loguru import logger
from modules.cost_estimator import CostEstimator, CostEstimatorRegistry
from modules.vel_command_generator import VelCommandGenerator, VelCommandGeneratorRegistry
from omegaconf import OmegaConf

from utils.cost_map import CostMap, CostMapRegistry
from utils.locomotion_model import LocoMotionBaseModel, LocoMotionModelRegistry
from utils.navigation import BasePlanner, PlannerRegistry
from utils.neural_map import NeuralMap, NeuralMapRegistry
from utils.position_controller import (PositionController,
                                       PositionControllerRegistry)

from helpers.state_recorder import StateRecorderRegistry, BaseStateRecorder
from helpers.dataset_collector import DatasetCollectorConfig, build_dataset_collector


def merge_config(cfg_obj, cfg_dict):
    schema = OmegaConf.structured(cfg_obj)
    merged_cfg = OmegaConf.merge(schema, cfg_dict)
    cfg_obj = OmegaConf.to_object(merged_cfg)
    return cfg_obj


def build_logger(cfg_omega, log_level="INFO"):
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    for handler in cfg["handlers"]:
        if handler["sink"] == "stdout":
            handler["sink"] = sys.stdout
            handler["level"] = log_level
        if handler["sink"] == "stderr":
            handler["sink"] = sys.stderr
            handler["level"] = log_level
    logger.configure(**cfg)
    return logger


def build_neural_map(nm_cfg_omega) -> NeuralMap:
    module_name = nm_cfg_omega.name
    cfg_omega = nm_cfg_omega.config
    num_envs = nm_cfg_omega.num_envs
    cls_, cfg_obj = NeuralMapRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    neural_map = cls_(num_envs, cfg_obj)
    if cfg_obj.load_dir is not None:
        neural_map.load(cfg_obj.load_dir)
    return neural_map


def build_cost_map(cm_cfg_omega) -> CostMap:
    module_name = cm_cfg_omega.name
    cfg_omega = cm_cfg_omega.config
    num_envs = cm_cfg_omega.num_envs
    cls_, cfg_obj = CostMapRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    cost_map = cls_(num_envs, cfg_obj)
    if cfg_obj.load_dir is not None:
        cost_map.load(cfg_obj.load_dir)
    return cost_map


def build_locomotion_model(model_cfg_omega) -> LocoMotionBaseModel:
    module_name = model_cfg_omega.name
    cfg_omega   = model_cfg_omega.config
    cls_, cfg_obj = LocoMotionModelRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    return cls_(cfg_local=cfg_obj)
    pass


def build_cost_estimator(model_cfg_omega, env_handle) -> CostEstimator:
    module_name = model_cfg_omega.name
    cfg_omega   = model_cfg_omega.config
    cls_, cfg_obj = CostEstimatorRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    return cls_(cfg_obj, env_handle=env_handle)


def build_position_controller(model_cfg_omega) -> PositionController:
    module_name = model_cfg_omega.name
    cfg_omega   = model_cfg_omega.config
    cls_, cfg_obj = PositionControllerRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    return cls_(cfg_obj)


def build_path_planner(module_cfg_omega) -> BasePlanner:
    module_name = module_cfg_omega.name
    cfg_omega = module_cfg_omega.config

    cls_, cfg_obj = PlannerRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    return cls_(cfg_obj)

def build_vel_cmd_generator(module_cfg_omega, context) -> VelCommandGenerator:
    module_name = module_cfg_omega.name
    cfg_omega = module_cfg_omega.config

    cls_, cfg_obj = VelCommandGeneratorRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    return cls_(cfg_obj, context=context)

def build_state_recorder(module_cfg_omega) -> BaseStateRecorder:
    module_name = module_cfg_omega.name
    cfg_omega = module_cfg_omega.config

    cls_, cfg_obj = StateRecorderRegistry.get(module_name)
    if cls_ is None:
        raise ValueError("Unknown Class")

    if cfg_omega is not None:
        cfg_obj = merge_config(cfg_obj, cfg_omega)

    return cls_(cfg_obj)


def build_runner_dataset_collector(cfg_omega):
    cfg = merge_config(DatasetCollectorConfig(), cfg_omega)
    return build_dataset_collector(cfg)



def merge_env_config(env_cfg_obj, env_cfg_omega):
    env_cfg_obj.seed                  = env_cfg_omega.seed

    env_cfg_obj.env.num_envs          = env_cfg_omega.env.num_envs
    env_cfg_obj.env.device            = env_cfg_omega.env.device
    env_cfg_obj.env.randomize_reset_heading \
                                      = OmegaConf.select(
                                          env_cfg_omega,
                                          "env.randomize_reset_heading",
                                          default=False,
                                      )
    commands_cfg_omega = OmegaConf.select(env_cfg_omega, "commands", default=None)
    if commands_cfg_omega is not None and hasattr(env_cfg_obj, "commands"):
        heading_command = OmegaConf.select(commands_cfg_omega, "heading_command", default=None)
        if heading_command is not None:
            env_cfg_obj.commands.heading_command = bool(heading_command)
        resampling_time = OmegaConf.select(commands_cfg_omega, "resampling_time", default=None)
        if resampling_time is not None:
            env_cfg_obj.commands.resampling_time = float(resampling_time)
    env_cfg_obj.terrain.curriculum    = env_cfg_omega.terrain.curriculum
    env_cfg_obj.terrain.num_rows      = env_cfg_omega.terrain.num_rows
    env_cfg_obj.terrain.num_cols      = env_cfg_omega.terrain.num_cols
    env_cfg_obj.terrain.terrains_yaml = env_cfg_omega.terrain.terrains_yaml
    env_cfg_obj.cost.scales.power     = env_cfg_omega.cost.scales.power
    env_cfg_obj.cost.scales.collision = env_cfg_omega.cost.scales.collision
    env_cfg_obj.cost.\
        visualize_collision_contacts \
                                      = env_cfg_omega.cost.visualize_collision_contacts
    env_cfg_obj.init_state.rot        = env_cfg_omega.init_state.rot
    return env_cfg_obj
