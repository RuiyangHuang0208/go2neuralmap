from .position_controller import PositionControllerConfig, PositionControllerCommand, PositionController, wrap_to_pi
from dataclasses import dataclass, field

import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
from loguru import logger

@dataclass
class PositionControllerRotationFirstConfig(PositionControllerConfig):
    pass


class PositionControllerRotationFirst(PositionController):
    def __init__(self, cfg: PositionControllerRotationFirstConfig):
        self.cfg = cfg
        self.logger = logger.bind(module_name="PositionController.RotationFirst")
        pass

    def compute_command(
        self,
        current_xy: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        target_xy: torch.Tensor,
        **kwargs,
    ) -> PositionControllerCommand:
        current_xy = current_xy.view(2)
        target_xy = target_xy.view(2)
        root_quat_xyzw = root_quat_xyzw.view(1, 4)

        position_error = target_xy - current_xy
        distance = torch.norm(position_error)

        forward_vec = current_xy.new_tensor([[1.0, 0.0, 0.0]])
        forward_dir = quat_apply(root_quat_xyzw, forward_vec)
        current_heading = torch.atan2(forward_dir[:, 1], forward_dir[:, 0]).squeeze(0)

        if distance <= self.cfg.goal_tolerance:
            zero_cmd = current_xy.new_zeros(2)
            return PositionControllerCommand(
                command_xy=zero_cmd,
                heading_target=current_heading,
                distance=distance,
                current_heading=current_heading,
                heading_error=current_heading.new_tensor(0.0),
            )

        heading_target = torch.atan2(position_error[1], position_error[0])
        heading_error = wrap_to_pi(heading_target - current_heading)
        abs_heading_error = torch.abs(heading_error)

        position_error_3d = current_xy.new_zeros((1, 3))
        position_error_3d[0, :2] = position_error
        position_error_body = quat_rotate_inverse(root_quat_xyzw, position_error_3d)[0, :2]

        if abs_heading_error >= self.cfg.turn_in_place_angle:
            zero_cmd = current_xy.new_zeros(2)
            return PositionControllerCommand(
                command_xy=zero_cmd,
                heading_target=heading_target,
                distance=distance,
                current_heading=current_heading,
                heading_error=heading_error,
            )

        forward_alignment = torch.clamp(torch.cos(heading_error), min=0.0, max=1.0)
        if abs_heading_error >= self.cfg.lateral_zero_angle:
            forward_scale = torch.clamp(forward_alignment, min=0.0, max=self.cfg.heading_slowdown)
            lateral_scale = current_heading.new_tensor(0.0)
        else:
            forward_scale = torch.clamp(forward_alignment, min=self.cfg.heading_slowdown, max=1.0)
            lateral_scale = torch.clamp(1.0 - abs_heading_error / self.cfg.lateral_zero_angle, min=0.0, max=1.0)

        command_xy = self.cfg.position_gain * position_error_body
        command_xy[0] = torch.clamp(command_xy[0], min=0.0, max=self.cfg.max_speed) * forward_scale
        command_xy[1] = torch.clamp(command_xy[1], min=-self.cfg.max_lateral_speed, max=self.cfg.max_lateral_speed) * lateral_scale

        planar_speed = torch.norm(command_xy)
        if planar_speed > self.cfg.max_speed:
            command_xy = command_xy * (self.cfg.max_speed / planar_speed)

        return PositionControllerCommand(
            command_xy=command_xy,
            heading_target=heading_target,
            distance=distance,
            current_heading=current_heading,
            heading_error=heading_error,
        )
