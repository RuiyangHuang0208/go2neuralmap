import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from typing import List, Tuple, Union


class LatentClsDataset(Dataset):
    def __init__(self, 
                 long_latent: np.ndarray,
                 short_latent: np.ndarray,
                 labels: np.ndarray,
                 mode: str="concat"
                 ):
        self.mode = mode

        self.long = torch.from_numpy(long_latent)
        self.short = torch.from_numpy(short_latent)
        self.y = torch.from_numpy(labels).long()


    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.mode == "long":
            x = self.long[idx]
        elif self.mode == "short":
            x = self.short[idx]
        elif self.mode == "concat":
            x = torch.cat([self.long[idx], self.short[idx]], dim=-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        y = self.y[idx]
        return x, y

def split_set(terrain_label):
    N = terrain_label.shape[0]
    all_idx = np.arange(N)

    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=terrain_label
    )
    return train_idx, test_idx


def _load_one_folder(folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load (long, short, labels) from a single dataset subfolder.
    Expected files:
      - inference_latents_long.npy
      - inference_latents_short.npy
      - inference_terrain_labels.npy
    """
    long_path = folder / "inference_latents_long.npy"
    short_path = folder / "inference_latents_short.npy"
    label_path = folder / "inference_terrain_labels.npy"

    if not (long_path.is_file() and short_path.is_file() and label_path.is_file()):
        raise FileNotFoundError(
            f"Missing required npy files in: {folder}\n"
            f"Expected: {long_path.name}, {short_path.name}, {label_path.name}"
        )

    long_latent = np.load(long_path, allow_pickle=True)
    short_latent = np.load(short_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)

    # 轻量一致性检查
    if len(long_latent) != len(short_latent) or len(long_latent) != len(labels):
        raise ValueError(
            f"Length mismatch in folder {folder}:\n"
            f"  long:  {len(long_latent)}\n"
            f"  short: {len(short_latent)}\n"
            f"  label: {len(labels)}"
        )

    return long_latent, short_latent, labels


def generate_dataset(
    dataset_root: Union[str, os.PathLike] = "/home/liwei/F/002Project/2025W/MasterThesis/legged_gym_go2/classifier/dataset",
    mode: str = "concat",
    unknown_label: int = 7,
    fake_ratio: float = 0.1,
    sort_folders: bool = True,
) -> Tuple[Subset, Subset]:
    """
    New dataset layout:
      dataset_root/
        2026-01-28_18-28-54/
          inference_latents_long.npy
          inference_latents_short.npy
          inference_terrain_labels.npy
        <more folders in future>...

    Behavior:
      - load all valid subfolders and concat them
      - add fake samples: all-zero long/short -> label=unknown_label
      - fake sample count = int(real_count * fake_ratio)
      - return (train_set, test_set) via split_set(labels)
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root not found or not a directory: {dataset_root}")

    subdirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    if sort_folders:
        subdirs.sort(key=lambda p: p.name)

    long_list: List[np.ndarray] = []
    short_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    valid_folders: List[Path] = []
    for folder in subdirs:
        try:
            l, s, y = _load_one_folder(folder)
        except FileNotFoundError:
            continue # Ignore in-complete directory
        long_list.append(l)
        short_list.append(s)
        label_list.append(y)
        valid_folders.append(folder)

    if len(long_list) == 0:
        raise FileNotFoundError(
            f"No valid dataset subfolders found under {dataset_root}.\n"
            f"Each subfolder must contain: inference_latents_long.npy, "
            f"inference_latents_short.npy, inference_terrain_labels.npy"
        )

    # concat 所有真实数据
    long_latent = np.concatenate(long_list, axis=0)
    short_latent = np.concatenate(short_list, axis=0)
    terrain_label = np.concatenate(label_list, axis=0)

    real_n = int(len(terrain_label))
    fake_n = int(real_n * float(fake_ratio))

    # 构造 fake data：shape 需要匹配单条样本的 latent shape
    if fake_n > 0:
        long_shape = (fake_n,) + tuple(long_latent.shape[1:])
        short_shape = (fake_n,) + tuple(short_latent.shape[1:])

        fake_long = np.zeros(long_shape, dtype=long_latent.dtype)
        fake_short = np.zeros(short_shape, dtype=short_latent.dtype)

        # label dtype 尽量保持一致（若不是整数，也会强制为 int64 更合理）
        if np.issubdtype(terrain_label.dtype, np.integer):
            fake_label = np.full((fake_n,), unknown_label, dtype=terrain_label.dtype)
        else:
            fake_label = np.full((fake_n,), unknown_label, dtype=np.int64)
            terrain_label = terrain_label.astype(np.int64, copy=False)

        long_latent = np.concatenate([long_latent, fake_long], axis=0)
        short_latent = np.concatenate([short_latent, fake_short], axis=0)
        terrain_label = np.concatenate([terrain_label, fake_label], axis=0)

    dataset = LatentClsDataset(
        long_latent=long_latent,
        short_latent=short_latent,
        labels=terrain_label,
        mode=mode,
    )

    train_idx, test_idx = split_set(terrain_label)
    train_set = Subset(dataset, train_idx.tolist())
    test_set = Subset(dataset, test_idx.tolist())
    return train_set, test_set
