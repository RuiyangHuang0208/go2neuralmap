import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    mode: str = "concat"
    short_dim: int = 32
    long_dim: int = 32
    output_dim: int = 7 + 1 # 7 type of terrains + 1 Unknown
    hidden_dims: list = (256, 512, 256)

    device: str = "cuda"

    lr: float = 1e-3
    num_epochs: int = 501
    batch_size = 32
    save_dir = "logs"

    dataset_dir: str = "/home/liwei/F/002Project/2025W/MasterThesis/legged_gym_go2/classifier/dataset"

    def calc_input_dim(self):
        if self.mode == "concat":
            return self.short_dim + self.long_dim
        elif self.mode == "short":
            return self.short_dim
        elif self.mode == "long":
            return self.long_dim
        else:
            raise ValueError(f"Undefined mode: {self.mode}")


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation="relu", if_normalize=False):
        super().__init__()
        prev = input_dim
        layers = []
        activation_func = self._get_activation(activation)
        self.if_normalize = if_normalize

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation_func)
            prev = h

        # output layer
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if self.if_normalize:
            x = (x - x.mean(dim=-1) / x.std(dim=-1))
        return self.net(x)

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "silu":
            return F.silu
        elif activation == "elu":
            return F.elu

    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs)
            return logits, preds
