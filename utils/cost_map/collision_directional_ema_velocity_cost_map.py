from __future__ import annotations

from dataclasses import dataclass

import torch

from .directional_ema_velocity_cost_map import (
    DirectionalEMAVelocityCostMap,
    DirectionalEMAVelocityCostMapCfg,
)


@dataclass
class CollisionDirectionalEMAVelocityCostMapCfg(DirectionalEMAVelocityCostMapCfg):
    impassable_cost: float = 1e6


class CollisionDirectionalEMAVelocityCostMap(DirectionalEMAVelocityCostMap):
    """Directional EMA velocity cost map with hard edge blocking from robot-frame collisions."""

    FRONT = 0
    LEFT = 1
    RIGHT = 2
    REAR = 3

    def __init__(self, num_envs: int, cfg: CollisionDirectionalEMAVelocityCostMapCfg):
        super().__init__(num_envs=num_envs, cfg=cfg)
        if cfg.impassable_cost < cfg.init_value:
            raise ValueError(
                f"impassable_cost must be >= init_value, got {cfg.impassable_cost} < {cfg.init_value}"
            )

    def update(self, current_pos: torch.Tensor, new_values: torch.Tensor, **kwargs) -> torch.Tensor:
        valid_mask = super().update(current_pos=current_pos, new_values=new_values, **kwargs)

        directional_collision = self._validate_directional_collision(
            kwargs.get("directional_collision"),
            "directional_collision",
        )
        base_quat = self._validate_base_quat(kwargs.get("base_quat"), "base_quat")

        collision_mask = torch.any(directional_collision, dim=-1)
        if not torch.any(collision_mask):
            return valid_mask

        x_indices, y_indices = self._xy_to_indices(current_pos)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        blocked_edge_ids = self._collision_flags_to_edge_ids(directional_collision, base_quat)
        impassable_cost = float(self.cfg.impassable_cost)

        for collision_dir_idx in range(4):
            dir_mask = collision_mask & directional_collision[:, collision_dir_idx]
            if not torch.any(dir_mask):
                continue

            self.maps[
                env_ids[dir_mask],
                x_indices[dir_mask],
                y_indices[dir_mask],
                blocked_edge_ids[dir_mask, collision_dir_idx],
            ] = impassable_cost

        return valid_mask | collision_mask

    def _validate_directional_collision(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} must be provided for {self.__class__.__name__}")
        value = value.to(device=self.device)
        if value.shape[0] != self.num_envs or value.shape[-1] != 4:
            raise ValueError(
                f"{name} must have shape ({self.num_envs}, 4), got {tuple(value.shape)}"
            )
        return value.to(dtype=torch.bool)

    def _validate_base_quat(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} must be provided for {self.__class__.__name__}")
        value = value.to(device=self.device, dtype=self.dtype)
        if value.shape[0] != self.num_envs or value.shape[-1] != 4:
            raise ValueError(
                f"{name} must have shape ({self.num_envs}, 4), got {tuple(value.shape)}"
            )
        return value

    def _collision_flags_to_edge_ids(
        self,
        directional_collision: torch.Tensor,
        base_quat: torch.Tensor,
    ) -> torch.Tensor:
        yaw = self._quat_to_yaw(base_quat)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        robot_frame_vectors = torch.tensor(
            [
                [1.0, 0.0],   # front
                [0.0, 1.0],   # left
                [0.0, -1.0],  # right
                [-1.0, 0.0],  # rear
            ],
            device=self.device,
            dtype=self.dtype,
        )
        world_vectors = torch.zeros(self.num_envs, 4, 2, device=self.device, dtype=self.dtype)
        world_vectors[:, :, 0] = (
            cos_yaw.unsqueeze(1) * robot_frame_vectors[:, 0]
            - sin_yaw.unsqueeze(1) * robot_frame_vectors[:, 1]
        )
        world_vectors[:, :, 1] = (
            sin_yaw.unsqueeze(1) * robot_frame_vectors[:, 0]
            + cos_yaw.unsqueeze(1) * robot_frame_vectors[:, 1]
        )

        edge_ids = torch.empty(self.num_envs, 4, device=self.device, dtype=torch.long)
        for collision_dir_idx in range(4):
            edge_ids[:, collision_dir_idx] = self._planar_vector_to_cardinal_channel(
                world_vectors[:, collision_dir_idx]
            )
        return edge_ids

    def _planar_vector_to_cardinal_channel(self, vectors: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(vectors[:, 0])
        abs_y = torch.abs(vectors[:, 1])
        use_x_axis = abs_x >= abs_y

        edge_ids = torch.empty(vectors.shape[0], device=self.device, dtype=torch.long)
        edge_ids[use_x_axis & (vectors[:, 0] >= 0.0)] = self.POS_X
        edge_ids[use_x_axis & (vectors[:, 0] < 0.0)] = self.NEG_X
        edge_ids[(~use_x_axis) & (vectors[:, 1] >= 0.0)] = self.POS_Y
        edge_ids[(~use_x_axis) & (vectors[:, 1] < 0.0)] = self.NEG_Y
        return edge_ids

    def _quat_to_yaw(self, quat_xyzw: torch.Tensor) -> torch.Tensor:
        x = quat_xyzw[:, 0]
        y = quat_xyzw[:, 1]
        z = quat_xyzw[:, 2]
        w = quat_xyzw[:, 3]
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(t3, t4)
