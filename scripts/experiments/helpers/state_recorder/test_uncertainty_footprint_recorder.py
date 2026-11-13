import torch

from scripts.experiments.helpers.state_recorder.uncertainty_footprint_recorder import (
    UncertaintyFootprintRecorder,
)


def _var_dict(step, current_xy, uncertainty):
    return {
        "step": step,
        "dt": 0.6,
        "current_xy": torch.tensor(current_xy),
        "uncertainty_map_vals": uncertainty,
        "uncertainty_map_metadata": {
            "x_range": (-2.0, 10.0),
            "y_range": (-2.0, 10.0),
            "least_count": 0.25,
        },
    }


def test_uncertainty_footprint_recorder_writes_periodic_channel_first_snapshot(tmp_path):
    recorder = UncertaintyFootprintRecorder(log_dir=str(tmp_path), snapshot_interval_seconds=1.0)
    uncertainty = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)

    recorder.handle(_var_dict(1, (1.0, 2.0), uncertainty))
    output_dir = tmp_path / "state_recorder" / "uncertainty_footprint"
    assert not output_dir.exists()

    recorder.handle(_var_dict(2, (3.0, 4.0), uncertainty + 1.0))

    snapshot = torch.load(output_dir / "snapshot_step_000002.pt", weights_only=True)
    assert snapshot["step"] == 2
    assert snapshot["elapsed_seconds"] == 1.2
    assert torch.equal(snapshot["uncertainty"], uncertainty + 1.0)
    assert torch.equal(snapshot["footprint_xy"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    metadata = torch.load(output_dir / "metadata.pt", weights_only=True)
    assert metadata["channel_names"] == ("pos_x", "neg_x", "pos_y", "neg_y")
    assert metadata["uncertainty_semantics"] == "aged"


def test_uncertainty_footprint_recorder_writes_final_unsampled_state(tmp_path):
    recorder = UncertaintyFootprintRecorder(log_dir=str(tmp_path), snapshot_interval_seconds=60.0)
    uncertainty = torch.ones(4, 2, 2)

    recorder.handle(_var_dict(7, (1.0, 2.0), uncertainty))
    recorder.save(tmp_path / "unused")

    snapshot_path = tmp_path / "state_recorder" / "uncertainty_footprint" / "snapshot_step_000007.pt"
    assert snapshot_path.exists()
