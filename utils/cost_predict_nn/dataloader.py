import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np


class CostMapDataset(Dataset):
    def __init__(self, input, label):
        self.input = input
        self.label = label

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.label[idx]

        return x, y

    def __len__(self):
        return len(self.input)

def split_set(dataset_length: int):
    N = dataset_length
    
    all_idx = np.arange(N)
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )
    return train_idx, test_idx
