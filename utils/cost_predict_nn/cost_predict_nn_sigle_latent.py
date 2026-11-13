from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    input_dim: int    = 64
    output_dim: int   = 1
    hidden_dims: list = field(default_factory=lambda: [128, 128])
    activation: str   = "relu" # Can be relu, silu, elu

    device: str       = "cuda"

    lr: float         = 1e-3
    num_epochs: int   = 501
    batch_size: str   = 32

    save_dir: str     = "logs"
    ckpt_path: str    = ""
    pass


class CostPredNNSingleLatentBuilder():
    def __init__(self):
        self.cfg = Config()

    def ckpt_path(self, ckpt_path):
        self.cfg.ckpt_path = ckpt_path
        return self

    def build(self):
        return CostPredNNSingleLatent(self.cfg)
    pass


class CostPredNNSingleLatent(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim
        hidden_dims = cfg.hidden_dims
        activation = cfg.activation

        prev = input_dim
        layers = []
        activation_func = self._get_activation(activation)

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation_func)
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "silu":
            return F.silu
        elif activation == "elu":
            return F.elu

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def load_weights(self):
        weights = torch.load(self.cfg.ckpt_path)
        self.load_state_dict(weights["model"])

    @staticmethod
    def builder():
        return CostPredNNSingleLatentBuilder()
