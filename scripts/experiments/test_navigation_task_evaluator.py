import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parent / "modules" / "vel_command_generator" / "navigation_task_evaluator.py"
SPEC = importlib.util.spec_from_file_location("navigation_task_evaluator", MODULE_PATH)
navigation_task_evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = navigation_task_evaluator
SPEC.loader.exec_module(navigation_task_evaluator)

NavigationTaskEvaluator = navigation_task_evaluator.NavigationTaskEvaluator
NavigationTaskEvaluatorConfig = navigation_task_evaluator.NavigationTaskEvaluatorConfig


class FakeEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.dt = 0.1
        self.task_goal_xy = torch.zeros((num_envs, 2), dtype=torch.float32)
        self.task_goal_valid = torch.zeros(num_envs, dtype=torch.bool)

    def resample_goals(self, env_ids):
        goals = torch.stack(
            [torch.tensor([float(env_id.item()) + 10.0, 1.0]) for env_id in env_ids],
            dim=0,
        )
        return type("ResetResult", (), {"goal_xy": goals})()


def test_navigation_task_evaluator_cycles_list_goals_on_success():
    env = FakeEnv(num_envs=1)
    evaluator = NavigationTaskEvaluator(
        env=env,
        cfg=NavigationTaskEvaluatorConfig(
            goal_source="list",
            goal_xy_list=[[1.0, 0.0], [2.0, 0.0]],
            goal_tolerance=0.2,
        ),
    )

    current_xy = torch.tensor([[0.0, 0.0]])
    evaluator.init_tasks(current_xy)
    assert torch.allclose(evaluator.goal_xy[0], torch.tensor([1.0, 0.0]))

    update = evaluator.update(current_xy=torch.tensor([[1.0, 0.0]]), dt=0.1)
    assert bool(update.terminal[0].item()) is True
    assert update.terminal_reason_by_env[0] == "success"
    assert update.info_by_env[0]["goal_index"] == 0

    evaluator.complete_terminals(update, current_xy=torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(evaluator.goal_xy[0], torch.tensor([2.0, 0.0]))


def test_navigation_task_evaluator_times_out_and_advances_goal():
    env = FakeEnv(num_envs=1)
    evaluator = NavigationTaskEvaluator(
        env=env,
        cfg=NavigationTaskEvaluatorConfig(
            goal_source="list",
            goal_xy_list=[[1.0, 0.0], [2.0, 0.0]],
            goal_tolerance=0.2,
            timeout_limit_steps=2,
        ),
    )

    current_xy = torch.tensor([[0.0, 0.0]])
    evaluator.init_tasks(current_xy)
    evaluator.update(current_xy=current_xy, dt=0.1)
    update = evaluator.update(current_xy=current_xy, dt=0.1)

    assert bool(update.timeout[0].item()) is True
    assert update.terminal_reason_by_env[0] == "timeout"
    assert update.info_by_env[0]["task_elapsed_steps"] == 2

    evaluator.complete_terminals(update, current_xy=current_xy)
    assert torch.allclose(evaluator.goal_xy[0], torch.tensor([2.0, 0.0]))
