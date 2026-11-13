import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parent / "vel_command_generator" / "vel_command_generator.py"
spec = importlib.util.spec_from_file_location("vel_command_generator_base", MODULE_PATH)
vel_command_generator_base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vel_command_generator_base)
normalize_vel_cmd_shape = vel_command_generator_base.normalize_vel_cmd_shape


def test_normalize_vel_cmd_shape_keeps_matching_2d_command():
    vel_cmd = torch.zeros((2, 4))

    normalized = normalize_vel_cmd_shape(vel_cmd, torch.Size((2, 4)))

    assert normalized is vel_cmd


def test_normalize_vel_cmd_shape_accepts_1d_command_for_single_env():
    vel_cmd = torch.tensor([1.0, 2.0, 3.0, 4.0])

    normalized = normalize_vel_cmd_shape(vel_cmd, torch.Size((1, 4)))

    assert torch.equal(normalized, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_normalize_vel_cmd_shape_rejects_1d_command_for_multi_env():
    with pytest.raises(ValueError, match="vel_cmd must have shape \\(2, 4\\), got \\(4,\\)"):
        normalize_vel_cmd_shape(torch.zeros(4), torch.Size((2, 4)))


def test_normalize_vel_cmd_shape_rejects_wrong_width():
    with pytest.raises(ValueError, match="vel_cmd must have shape \\(1, 4\\), got \\(3,\\)"):
        normalize_vel_cmd_shape(torch.zeros(3), torch.Size((1, 4)))
