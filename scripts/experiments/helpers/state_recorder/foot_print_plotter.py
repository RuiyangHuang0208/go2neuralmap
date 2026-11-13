from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, axis
from loguru import logger
import torch

from .sub_recorder import SubPlotter


class FootPrint(SubPlotter):
    def __init__(self, ax: Axes):
        self.ax = ax
        self.ax.set_xlim(left=-2.0, right=10.0)
        self.ax.set_ylim(bottom=-2.0, top=10.0)
        self.ax.set_title("FootPrint")
        self.foot_print = []
        self.data = torch.empty((0, 2), dtype=torch.float32)
        self.scatter = None
        pass

    def draw(self, var_dict):
        current_xy = var_dict.get("current_xy")
        if current_xy is None:
            return
        self.foot_print.append(current_xy)

        data = torch.stack(self.foot_print, dim=0)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)
        self.data = data
        if self.scatter is None:
            z = torch.arange(data.shape[0])  # point_idx，used for color gradient
            self.scatter = self.ax.scatter(data[:, 0], data[:, 1], c=z, cmap='viridis', s=10, alpha=0.8)
            return

        self.scatter.set_offsets(data)

        colors = plt.cm.viridis(torch.linspace(0, 1, data.shape[0]))
        self.scatter.set_facecolors(colors)

    def save(self, save_dir=""):
        torch.save(
            {
                "foot_print": self.data
            },
            f"{save_dir}/foot_print.pt",
        )
