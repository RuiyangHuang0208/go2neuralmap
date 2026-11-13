from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
import torch
import numpy as np
from loguru import logger
from legged_gym.envs.go2.go2_nm_env import GO2NM

from .vel_command_generator import (VelCommandGenerator,
                                    VelCommandGeneratorConfig)


def quaternion_yaw(quat_xyzw):
    """
    计算四元数相对于 [0, 0, 0, 1] 的yaw角（绕Z轴旋转角度）
    
    参数:
        quat_xyzw: 四元数，格式为 [x, y, z, w]
    
    返回:
        yaw: 绕Z轴的旋转角度（弧度），范围 [-π, π]
    """
    quat = np.array(quat_xyzw, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return yaw


@dataclass
class KeyboardInputConfig(VelCommandGeneratorConfig):
    enable_robot_reset: bool = True
    robot_reset_key: str = "r"


class KeyboardInput(VelCommandGenerator):
    def __init__(self,
                 cfg: KeyboardInputConfig = KeyboardInputConfig(),
                 context: Optional = None, **kwargs):
        from utils.user_input import LinuxKeyboardController

        self.kbd_controller = LinuxKeyboardController()
        self.context = context
        self.env: GO2NM = self.context.env
        self.cfg = cfg
        self.heading = quaternion_yaw(np.array(self.env.cfg.init_state.rot))
        self.last_heading_state = self.heading
        pass

    def compute_vel_cmd(self, **kwargs):
        cmd = self.kbd_controller.get_command()

        x_val = cmd[0]
        y_val = cmd[1]
        rot = cmd[2]

        if rot > 0 and rot != self.last_heading_state:
            self.heading += 0.1 * torch.pi
        elif rot < 0 and rot != self.last_heading_state:
            self.heading -= 0.1 * torch.pi

        self.last_heading_state = rot

        # heading = np.clip(heading, -np.pi, np.pi)
        if self.heading < -torch.pi:
            self.heading += 2 * torch.pi
        if self.heading > torch.pi:
            self.heading -= 2 * torch.pi

        env_cmd = torch.zeros(4, dtype=torch.float, device=self.env.device)
        env_cmd[0] = x_val
        env_cmd[1] = y_val
        env_cmd[3] = self.heading
        return env_cmd

    def init_state(self, **kwargs):
        pass

    def consume_robot_reset_request(self) -> bool:
        """Return a one-shot manual robot-reset request, if enabled."""
        if not self.cfg.enable_robot_reset:
            return False
        return self.kbd_controller.consume_key_press(self.cfg.robot_reset_key)

    def retrieve_markers(self) -> List:
        return []

    def get_info(self) -> Dict:
        return {}
