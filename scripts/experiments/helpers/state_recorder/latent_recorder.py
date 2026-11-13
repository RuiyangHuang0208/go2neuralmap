from copy import deepcopy

import torch
from loguru import logger

from .sub_recorder import SubRecorder


class LatentRecorder(SubRecorder):

    def __init__(self, **kwargs):
        self.long_latents = []
        self.short_latents = []
        self.positions = []

    def handle(self, var_dict):
        curr_latent_long: torch.Tensor = var_dict.get("long_latent")
        curr_latent_short: torch.Tensor = var_dict.get("short_latent")
        curr_xy = var_dict.get("current_xy")

        if curr_latent_long is None or curr_latent_short is None:
            raise ValueError

        self.long_latents.append(curr_latent_long.clone())
        self.short_latents.append(curr_latent_short.clone())
        self.positions.append(curr_xy)

    def save(self, save_dir):
        longs = torch.stack(self.long_latents)
        shorts = torch.stack(self.short_latents)
        positions = torch.stack(self.positions)
        torch.save({
            "long_latents": longs,
            "short_latents": shorts,
            "positions": positions,
        }, f"{save_dir}/latents.pt")
        pass
