from copy import deepcopy

import torch
from loguru import logger

from .sub_recorder import SubRecorder


class CostRecorder(SubRecorder):

    def __init__(self, **kwargs):
        self.costs = []
        pass

    def handle(self, var_dict):
        curr_cost: torch.Tensor = var_dict.get("current_cost") # of shape (num_envs,)
        if curr_cost is None:
            raise ValueError
        self.costs.append(curr_cost.clone())

    def save(self, save_dir):
        costs = torch.stack(self.costs)
        torch.save({
            "costs": costs
        }, f"{save_dir}/costs.pt")
        pass
