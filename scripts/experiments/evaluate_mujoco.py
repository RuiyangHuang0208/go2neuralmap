#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import OmegaConf

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from builder import build_logger  # noqa: E402
from mujoco_runtime.env_adapter import MuJoCoGo2EnvConfig  # noqa: E402
from mujoco_runtime.runner import MuJoCoEvaluationConfig, MuJoCoEvaluationRunner  # noqa: E402
from mujoco_runtime.scene_builder import (  # noqa: E402
    build_mujoco_scene_from_config,
    build_mujoco_scene_from_terrain_yaml,
)


MAIN_LOGGER = logger.bind(module_name="MuJoCoMain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the Go2 stack in MuJoCo.")
    parser.add_argument("--my_exp_name", type=str, default="go2nm", help="Experiment config under configs/exp.")
    parser.add_argument(
        "--xml_path",
        type=str,
        default="legged_gym/resources/robots/go2/mujoco/scene.xml",
        help="MuJoCo XML scene path.",
    )
    parser.add_argument(
        "--robot_xml",
        type=str,
        default="legged_gym/resources/robots/go2/mujoco/go2.xml",
        help="Go2 MuJoCo robot XML used when --terrain_yaml generates a scene.",
    )
    parser.add_argument("--terrain_yaml", type=str, default=None, help="Terrain YAML used to generate a MuJoCo scene.")
    parser.add_argument(
        "--mujoco_scene_config",
        type=str,
        default=None,
        help="MuJoCo-specific terrain scene config. Requires --terrain_yaml for start/goal semantics.",
    )
    parser.add_argument("--terrain_rows", type=int, default=None, help="Override generated terrain grid rows.")
    parser.add_argument("--terrain_cols", type=int, default=None, help="Override generated terrain grid cols.")
    parser.add_argument("--terrain_scene_dir", type=str, default=None, help="Optional output dir for generated MuJoCo scene.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override.")
    parser.add_argument("--render", action="store_true", help="Open MuJoCo passive viewer.")
    parser.add_argument("--realtime", action="store_true", help="Sleep to approximately match control_dt.")
    parser.add_argument("--render_fps", type=float, default=30.0, help="Viewer sync rate when rendering.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Stop after this many terminal navigation episodes. Defaults to 1 headless, or 0 with --render to run max_steps.",
    )
    parser.add_argument("--max_steps", type=int, default=2500)
    parser.add_argument(
        "--command_mode",
        choices=("fixed", "point_to_point", "config"),
        default="fixed",
        help="fixed ignores configured command generator; point_to_point injects a simple goal task; config uses exp config.",
    )
    parser.add_argument("--fixed_command", type=float, nargs=3, default=(0.4, 0.0, 0.0))
    parser.add_argument("--goal_xy", type=float, nargs=2, default=(2.0, 0.0))
    parser.add_argument("--no_update_neural_map", action="store_true", help="Disable neural map updates during MuJoCo rollout.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--deactivate_state_recorder", action="store_true", default=False)
    return parser.parse_args()


def load_configs(args: argparse.Namespace):
    commit_hash = os.environ.get("COMMIT_HASH", "mujoco")
    time_now = datetime.now()
    curr_log_dir_name = os.path.join(
        f"{time_now.strftime('%Y-%m-%d')}_{commit_hash}",
        f"{args.my_exp_name}_mujoco_{time_now.strftime('%H-%M-%S')}",
    )
    with hydra.initialize(config_path="configs", version_base="1.1"):
        cfg_omega = hydra.compose(
            config_name="base_config",
            overrides=[
                f"+exp={args.my_exp_name}",
                f"helpers.state_recorder.name={'null_recorder' if args.deactivate_state_recorder else 'state_recorder'}",
                f"+curr_log_dir_name={curr_log_dir_name}",
            ],
        )
    OmegaConf.resolve(cfg_omega)
    return cfg_omega


def main() -> int:
    args = parse_args()
    episodes = 0 if args.episodes is None and args.render else 1 if args.episodes is None else int(args.episodes)
    cfg_omega = load_configs(args)
    if args.terrain_yaml is not None:
        cfg_omega.env.terrain.terrains_yaml = str(args.terrain_yaml)
    if args.terrain_rows is not None:
        cfg_omega.env.terrain.num_rows = int(args.terrain_rows)
    if args.terrain_cols is not None:
        cfg_omega.env.terrain.num_cols = int(args.terrain_cols)
    if args.command_mode == "point_to_point":
        cfg_omega.modules.vel_command_generator.name = "point_to_point_navigation"
        cfg_omega.modules.vel_command_generator.config = {
            "goal_source": "list",
            "goal_xy_list": [[float(args.goal_xy[0]), float(args.goal_xy[1])]],
            "goal_tolerance": 0.35,
            "timeout_limit_steps": int(args.max_steps),
            "timeout_action": "task_reset",
            "reset_env_flag": False,
            "draw_goal_marker": False,
        }
    build_logger(cfg_omega.helpers.Logger, args.log_level)

    terrain_scene = None
    xml_path = args.xml_path
    if args.mujoco_scene_config is not None:
        if args.terrain_yaml is None:
            raise ValueError("--mujoco_scene_config requires --terrain_yaml so starts/goals keep terrain-loader semantics.")
        terrain_scene = build_mujoco_scene_from_config(
            terrain_cfg_omega=cfg_omega.env.terrain,
            mujoco_scene_config_path=args.mujoco_scene_config,
            robot_xml_path=args.robot_xml,
            work_dir=args.terrain_scene_dir,
            device=args.device or str(OmegaConf.select(cfg_omega, "training.device", default="cuda")),
        )
        xml_path = str(terrain_scene.xml_path)
        MAIN_LOGGER.info("Generated MuJoCo configured scene at {}", terrain_scene.xml_path)
    elif args.terrain_yaml is not None:
        terrain_scene = build_mujoco_scene_from_terrain_yaml(
            terrain_cfg_omega=cfg_omega.env.terrain,
            robot_xml_path=args.robot_xml,
            work_dir=args.terrain_scene_dir,
            device=args.device or str(OmegaConf.select(cfg_omega, "training.device", default="cuda")),
        )
        xml_path = str(terrain_scene.xml_path)
        MAIN_LOGGER.info("Generated MuJoCo terrain scene at {}", terrain_scene.xml_path)

    mujoco_cfg = MuJoCoGo2EnvConfig(
        xml_path=xml_path,
        device=args.device or str(OmegaConf.select(cfg_omega, "training.device", default="cuda")),
        render=bool(args.render),
        realtime=bool(args.realtime or args.render),
        render_fps=float(args.render_fps),
        goal_xy=tuple(float(v) for v in args.goal_xy),
    )
    eval_cfg = MuJoCoEvaluationConfig(
        episodes=episodes,
        max_steps=int(args.max_steps),
        command_mode=str(args.command_mode),
        fixed_command=tuple(float(v) for v in args.fixed_command),
        update_neural_map=not bool(args.no_update_neural_map),
        output=args.output,
    )
    MAIN_LOGGER.info("MuJoCo config: {}", asdict(mujoco_cfg))
    MAIN_LOGGER.info("Evaluation config: {}", asdict(eval_cfg))
    runner = MuJoCoEvaluationRunner(
        cfg_omega,
        mujoco_cfg,
        eval_cfg,
        terrain_loader=terrain_scene.terrain_loader if terrain_scene is not None else None,
    )
    summary = runner.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
