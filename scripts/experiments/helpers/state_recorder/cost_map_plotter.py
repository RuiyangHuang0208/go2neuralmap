from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, axis
from loguru import logger
import torch
from loguru import logger
from .sub_recorder import SubPlotter

class CostMap(SubPlotter):
    """
    Cost map needed to be
    cost_map_vals = cost_map.maps[0, :, :, 0].detach().cpu()
    """

    def __init__(self, ax: Axes):
        self.ax = ax
        self.ax.set_title("CostMapHeatMap")
        self.im = None
        self.cost_map_vals = torch.empty((0, 0, 0), dtype=torch.float32)
        self.logger = logger.bind(module_name="StateRecorder.CostMapPlotter")
        pass

    def draw(self, var_dict):
        cost_map_vals = var_dict.get("cost_map_vals")
        if cost_map_vals is None:
            return
        if self.im is None:
            self.im = self.ax.imshow(cost_map_vals.mean(dim=-1), cmap="hot")
            self.im.set_clim(vmin=40.0, vmax=75.0)
            plt.colorbar(self.im)
        self.im.set_data(cost_map_vals.mean(dim=-1))
        self.ax.autoscale_view()
        self.cost_map_vals = cost_map_vals

    def save(self, save_dir):
        torch.save(
            {
                "cost_map_vals": self.cost_map_vals
            },
            f"{save_dir}/cost_map_vals.pt"
        )
        self.logger.info(f"Cost map values saved to {save_dir}/cost_map_vals.pt")
        return
