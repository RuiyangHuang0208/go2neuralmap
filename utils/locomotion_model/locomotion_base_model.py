import torch.nn as nn
from dataclasses import dataclass, field
from abc import abstractmethod

@dataclass
class LocoMotionBaseModelCfg:
    pass


class LocoMotionBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = {}

    @abstractmethod
    def compute_latents(self, obs):
        pass

    @abstractmethod
    def act(self):
        pass


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
