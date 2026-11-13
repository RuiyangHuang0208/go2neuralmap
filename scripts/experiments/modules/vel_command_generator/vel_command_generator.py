from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class VelCommandGeneratorConfig:
    pass


def normalize_vel_cmd_shape(vel_cmd: torch.Tensor, expected_shape: torch.Size) -> torch.Tensor:
    expected_shape = torch.Size(expected_shape)
    if vel_cmd.shape == expected_shape:
        return vel_cmd
    if (
        len(expected_shape) == 2
        and expected_shape[0] == 1
        and vel_cmd.shape == torch.Size((expected_shape[1],))
    ):
        return vel_cmd.view(1, expected_shape[1])
    raise ValueError(
        f"vel_cmd must have shape {tuple(expected_shape)}, got {tuple(vel_cmd.shape)}"
    )


class VelCommandGenerator(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def compute_vel_cmd(self, **kwargs):
        pass

    @abstractmethod
    def init_state(self, **kwargs):
        pass

    @abstractmethod
    def retrieve_markers(self) -> List:
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        pass
