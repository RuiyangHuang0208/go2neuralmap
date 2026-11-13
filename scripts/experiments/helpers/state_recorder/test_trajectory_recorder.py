import torch

from scripts.experiments.helpers.state_recorder.trajectory_recorder import TrajectoryRecorder


def make_var_dict(
    *,
    current_xy=(1.0, 2.0),
    current_grid_xy=(1, 2),
    traj_cost=3.0,
    reached_goal_flag=False,
    done=False,
):
    return {
        "vel_cmd_generator_info": {
            "current_xy": torch.tensor(current_xy),
            "current_grid_xy": current_grid_xy,
            "traj_cost": traj_cost,
            "reached_goal_flag": reached_goal_flag,
        },
        "reset_buf": torch.tensor([done]),
    }


def test_trajectory_recorder_save_without_samples_writes_empty_lists(tmp_path):
    recorder = TrajectoryRecorder()

    recorder.save(tmp_path)

    saved = torch.load(tmp_path / "trajectories.pt", weights_only=False)
    assert saved["trajs"] == []
    assert saved["grid_trajs"] == []
    assert saved["traj_costs"] == []


def test_trajectory_recorder_save_includes_in_progress_trajectory(tmp_path):
    recorder = TrajectoryRecorder()

    recorder.handle(make_var_dict())
    recorder.save(tmp_path)

    saved = torch.load(tmp_path / "trajectories.pt", weights_only=False)
    assert len(saved["trajs"]) == 1
    assert torch.equal(saved["trajs"][0], torch.tensor([[1.0, 2.0]]))
    assert saved["grid_trajs"] == []
    assert saved["traj_costs"] == [None]


def test_trajectory_recorder_save_does_not_mutate_recorded_trajectories(tmp_path):
    recorder = TrajectoryRecorder()
    recorder.handle(make_var_dict())

    recorder.save(tmp_path)
    recorder.save(tmp_path)

    saved = torch.load(tmp_path / "trajectories.pt", weights_only=False)
    assert len(saved["trajs"]) == 1


def test_trajectory_recorder_handles_navigation_info_without_grid_coordinates(tmp_path):
    recorder = TrajectoryRecorder()
    var_dict = make_var_dict()
    del var_dict["vel_cmd_generator_info"]["current_grid_xy"]

    recorder.handle(var_dict)
    recorder.save(tmp_path)

    saved = torch.load(tmp_path / "trajectories.pt", weights_only=False)
    assert len(saved["trajs"]) == 1
    assert saved["grid_trajs"] == []


def test_trajectory_recorder_handles_missing_navigation_info():
    recorder = TrajectoryRecorder()

    recorder.handle({"vel_cmd_generator_info": None, "reset_buf": torch.tensor([False])})

    assert recorder.curr_traj == []
    assert recorder.curr_grid_traj == []
