import json

import torch

from scripts.experiments.helpers.state_recorder.navigation_metrics_recorder import NavigationMetricsRecorder


def make_var_dict(
    *,
    current_xy,
    goal_xy=(1.0, 0.0),
    reached_goal_flag=False,
    task_info=None,
    done=False,
    collision_flags=None,
    dt=0.1,
):
    if collision_flags is None:
        collision_flags = torch.tensor([[False, False, False, False]])
    vel_cmd_generator_info = {
        "current_xy": torch.tensor(current_xy),
        "goal_xy": torch.tensor(goal_xy),
        "reached_goal_flag": reached_goal_flag,
    }
    if task_info:
        vel_cmd_generator_info.update(task_info)
    return {
        "vel_cmd_generator_info": vel_cmd_generator_info,
        "reset_buf": torch.tensor([done]),
        "collision_flags": collision_flags,
        "dt": dt,
    }


def test_navigation_metrics_recorder_records_success_time_collision_and_spl(tmp_path):
    recorder = NavigationMetricsRecorder()

    recorder.handle(make_var_dict(current_xy=(0.0, 0.0)))
    recorder.handle(
        make_var_dict(
            current_xy=(1.0, 0.0),
            reached_goal_flag=True,
            collision_flags=torch.tensor([[True, False, True, False]]),
        )
    )
    recorder.save(tmp_path)

    saved = json.loads((tmp_path / "navigation_metrics.json").read_text())
    assert saved["summary"]["episode_count"] == 1
    assert saved["summary"]["success_count"] == 1
    assert saved["summary"]["success_rate"] == 1.0
    assert saved["summary"]["time_to_goal_mean_s"] == 0.2
    assert saved["summary"]["collision_count_mean"] == 1.0
    assert saved["summary"]["success_per_length_mean"] == 1.0
    assert saved["episodes"][0]["directional_collision_counts"] == [1, 0, 1, 0]


def test_navigation_metrics_recorder_records_failed_done_episode(tmp_path):
    recorder = NavigationMetricsRecorder()

    recorder.handle(make_var_dict(current_xy=(0.0, 0.0)))
    recorder.handle(make_var_dict(current_xy=(0.5, 0.0), done=True))
    recorder.save(tmp_path)

    saved = json.loads((tmp_path / "navigation_metrics.json").read_text())
    episode = saved["episodes"][0]
    assert episode["success"] is False
    assert episode["terminal_reason"] == "done"
    assert episode["time_to_goal_s"] is None
    assert episode["success_per_length"] == 0.0


def test_navigation_metrics_recorder_records_task_timeout_duration(tmp_path):
    recorder = NavigationMetricsRecorder()

    recorder.handle(make_var_dict(current_xy=(0.0, 0.0)))
    recorder.handle(
        make_var_dict(
            current_xy=(0.2, 0.0),
            task_info={
                "task_terminal_flag": True,
                "task_success_flag": False,
                "task_timeout_flag": True,
                "task_terminal_reason": "timeout",
                "task_elapsed_time_s": 3.5,
            },
        )
    )
    recorder.save(tmp_path)

    saved = json.loads((tmp_path / "navigation_metrics.json").read_text())
    episode = saved["episodes"][0]
    assert episode["success"] is False
    assert episode["terminal_reason"] == "timeout"
    assert episode["task_duration_s"] == 3.5
    assert saved["summary"]["timeout_count"] == 1
    assert saved["summary"]["timeout_rate"] == 1.0
    assert saved["summary"]["task_duration_mean_s"] == 3.5
