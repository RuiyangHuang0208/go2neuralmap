import torch

from utils.reset_events import ResetResult


def test_reset_result_carries_goal_resample_payload():
    result = ResetResult(
        env_ids=torch.tensor([0, 2]),
        reset_type="goal",
        terrain_ids=torch.tensor([1, 1]),
        start_xy=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        goal_xy=torch.tensor([[2.0, 0.0], [3.0, 1.0]]),
        root_local_pos=torch.tensor([[0.5, 0.0, 0.2], [1.5, 1.0, 0.3]]),
    )

    assert result.reset_type == "goal"
    assert torch.equal(result.env_ids, torch.tensor([0, 2]))
    assert result.goal_xy is not None
    assert result.goal_xy.shape == (2, 2)
    assert result.done_mask is None

