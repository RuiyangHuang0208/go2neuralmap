from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_LATENTS_PATH = (
    "logs/2026-06-06_b434b5c/demo_plane_kbd_4_22-01-50/"
    "state_recorder/latents.pt"
)

LABEL_TO_ID = {"plane": 0, "stair": 1, "slope": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def torch_load(path: str | Path, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_latents_path(latents_path: str | Path) -> Path:
    latents_path = Path(latents_path)
    if latents_path.is_file():
        return latents_path
    if latents_path.is_dir():
        candidates = [
            latents_path / "latents.pt",
            latents_path / "state_recorder" / "latents.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"{latents_path} is a directory, but no latents.pt was found at "
            f"{candidates[0]} or {candidates[1]}"
        )
    raise FileNotFoundError(f"latents path does not exist: {latents_path}")


@dataclass
class TrainResult:
    checkpoint_path: Path
    train_accuracy: float
    val_accuracy: float


class TerrainMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 128),
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _squeeze_latent(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D] or [N, 1, D], got {tuple(x.shape)}")
    return x.float()


def load_latent_tensors(latents_path: str | Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents_path = resolve_latents_path(latents_path)
    data = torch_load(latents_path, map_location="cpu")
    required_keys = ("long_latents", "short_latents", "positions")
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"{latents_path} is missing required keys: {missing_keys}")

    long_latents = _squeeze_latent(data["long_latents"], "long_latents")
    short_latents = _squeeze_latent(data["short_latents"], "short_latents")
    positions = data["positions"]
    if positions.ndim == 3 and positions.shape[1] == 1:
        positions = positions.squeeze(1)
    if positions.ndim != 2 or positions.shape[-1] < 2:
        raise ValueError(f"positions must have shape [N, 2+] or [N, 1, 2+], got {tuple(positions.shape)}")
    positions = positions.float()

    sample_count = long_latents.shape[0]
    if short_latents.shape[0] != sample_count or positions.shape[0] != sample_count:
        raise ValueError(
            "long_latents, short_latents, and positions must have the same sample count: "
            f"{long_latents.shape[0]}, {short_latents.shape[0]}, {positions.shape[0]}"
        )
    return long_latents, short_latents, positions


def labels_from_positions(positions: torch.Tensor) -> torch.Tensor:
    labels = torch.empty(positions.shape[0], dtype=torch.long)
    x = positions[:, 0]
    y = positions[:, 1]

    plane_mask = (x > 6.0) | (x < 4.0)
    stair_mask = (~plane_mask) & (y < 4.0)
    slope_mask = (~plane_mask) & (~stair_mask)

    labels[plane_mask] = LABEL_TO_ID["plane"]
    labels[stair_mask] = LABEL_TO_ID["stair"]
    labels[slope_mask] = LABEL_TO_ID["slope"]
    return labels


def build_concat_dataset(latents_path: str | Path) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    long_latents, short_latents, positions = load_latent_tensors(latents_path)
    x = torch.cat([long_latents, short_latents], dim=-1)
    y = labels_from_positions(positions)
    counts = {
        ID_TO_LABEL[int(label_id)]: int((y == label_id).sum().item())
        for label_id in sorted(ID_TO_LABEL)
    }
    return x, y, counts


def split_indices(
    labels: torch.Tensor,
    val_ratio: float,
    seed: int,
    stratified: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if not stratified:
        perm = torch.randperm(labels.shape[0], generator=generator)
        val_count = max(1, int(round(labels.shape[0] * val_ratio)))
        return perm[val_count:], perm[:val_count]

    train_parts = []
    val_parts = []
    for label_id in labels.unique(sorted=True):
        idx = torch.where(labels == label_id)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        val_count = max(1, int(round(idx.numel() * val_ratio)))
        if idx.numel() <= 1:
            train_parts.append(idx)
            continue
        val_count = min(val_count, idx.numel() - 1)
        val_parts.append(idx[:val_count])
        train_parts.append(idx[val_count:])

    train_idx = torch.cat(train_parts)
    val_idx = torch.cat(val_parts) if val_parts else torch.empty(0, dtype=torch.long)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=generator)]
    return train_idx, val_idx


def standardize_train_val(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return float((preds == labels).float().mean().item())


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[float, torch.Tensor]:
    model.eval()
    preds = []
    correct = 0
    total = 0
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            batch_preds = torch.argmax(logits, dim=-1)
            correct += int((batch_preds == yb).sum().item())
            total += int(yb.numel())
            preds.append(batch_preds.cpu())
    pred_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    return correct / max(total, 1), pred_tensor


def train_model(args: argparse.Namespace) -> TrainResult:
    torch.manual_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        device = "cpu"

    x, y, counts = build_concat_dataset(args.latents_path)
    train_idx, val_idx = split_indices(y, args.val_ratio, args.seed, stratified=not args.no_stratify)
    train_x = x[train_idx]
    train_y = y[train_idx]
    val_x = x[val_idx]
    val_y = y[val_idx]
    train_x, val_x, mean, std = standardize_train_val(train_x, val_x)

    hidden_dims = tuple(int(dim) for dim in args.hidden_dims.split(",") if dim.strip())
    model = TerrainMLP(
        input_dim=train_x.shape[-1],
        hidden_dims=hidden_dims,
        output_dim=len(LABEL_TO_ID),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Loaded {x.shape[0]} samples from {args.latents_path}")
    print(f"Input dim: {x.shape[-1]}")
    print(f"Terrain counts: {json.dumps(counts, sort_keys=True)}")
    print(f"Train/val split: {len(train_idx)}/{len(val_idx)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_acc, _ = evaluate(model, train_x, train_y, args.batch_size, device)
            val_acc, val_preds = evaluate(model, val_x, val_y, args.batch_size, device)
            avg_loss = total_loss / max(len(train_idx), 1)
            print(
                f"epoch {epoch:04d}/{args.epochs} "
                f"loss={avg_loss:.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

    train_acc, _ = evaluate(model, train_x, train_y, args.batch_size, device)
    val_acc, val_preds = evaluate(model, val_x, val_y, args.batch_size, device)
    cm = confusion_matrix(val_preds, val_y, num_classes=len(LABEL_TO_ID))
    print("Validation confusion matrix rows=true, cols=pred:")
    print(cm.tolist())

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": int(train_x.shape[-1]),
        "hidden_dims": list(hidden_dims),
        "dropout": float(args.dropout),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "mean": mean,
        "std": std,
        "terrain_rule": "plane if x < 4.0 or x > 6.0; stair if 4.0 <= x <= 6.0 and y < 4.0; slope otherwise",
        "source_latents_path": str(args.latents_path),
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_confusion_matrix": cm,
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")
    return TrainResult(save_path, train_acc, val_acc)


class TerrainLatentClassifier:
    def __init__(
        self,
        model: TerrainMLP,
        mean: torch.Tensor,
        std: torch.Tensor,
        id_to_label: Dict[int, str],
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.mean = mean.to(device)
        self.std = std.to(device)
        self.id_to_label = {int(k): v for k, v in id_to_label.items()}
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "cpu") -> "TerrainLatentClassifier":
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        model = TerrainMLP(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dims=checkpoint["hidden_dims"],
            output_dim=len(checkpoint["label_to_id"]),
            dropout=float(checkpoint.get("dropout", 0.0)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(
            model=model,
            mean=checkpoint["mean"].float(),
            std=checkpoint["std"].float(),
            id_to_label=checkpoint["id_to_label"],
            device=device,
        )

    def _prepare_concat(self, concat_latents: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(concat_latents):
            concat_latents = torch.as_tensor(concat_latents)
        concat_latents = concat_latents.float()
        if concat_latents.ndim == 1:
            concat_latents = concat_latents.unsqueeze(0)
        if concat_latents.ndim != 2:
            raise ValueError(f"concat_latents must have shape [D] or [N, D], got {tuple(concat_latents.shape)}")
        return ((concat_latents.to(self.device) - self.mean) / self.std).float()

    def predict_proba_concat(self, concat_latents: torch.Tensor) -> torch.Tensor:
        x = self._prepare_concat(concat_latents)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=-1).cpu()

    def predict_concat(self, concat_latents: torch.Tensor) -> List[str]:
        probs = self.predict_proba_concat(concat_latents)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        return [self.id_to_label[int(pred_id)] for pred_id in pred_ids]

    def predict_proba(self, long_latents: torch.Tensor, short_latents: torch.Tensor) -> torch.Tensor:
        long_latents = _squeeze_latent(torch.as_tensor(long_latents), "long_latents")
        short_latents = _squeeze_latent(torch.as_tensor(short_latents), "short_latents")
        return self.predict_proba_concat(torch.cat([long_latents, short_latents], dim=-1))

    def predict(self, long_latents: torch.Tensor, short_latents: torch.Tensor) -> List[str]:
        probs = self.predict_proba(long_latents, short_latents)
        pred_ids = torch.argmax(probs, dim=-1).tolist()
        return [self.id_to_label[int(pred_id)] for pred_id in pred_ids]


def predict_from_file(args: argparse.Namespace) -> None:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        device = "cpu"

    classifier = TerrainLatentClassifier.load(args.checkpoint, device=device)
    long_latents, short_latents, positions = load_latent_tensors(args.latents_path)
    concat_latents = torch.cat([long_latents, short_latents], dim=-1)
    probs = classifier.predict_proba_concat(concat_latents)
    pred_ids = torch.argmax(probs, dim=-1)
    true_ids = labels_from_positions(positions)

    pred_counts = {
        classifier.id_to_label[label_id]: int((pred_ids == label_id).sum().item())
        for label_id in sorted(classifier.id_to_label)
    }
    accuracy = float((pred_ids == true_ids).float().mean().item())
    cm = confusion_matrix(pred_ids, true_ids, num_classes=len(classifier.id_to_label))
    print(f"Predicted {pred_ids.numel()} samples from {args.latents_path}")
    print(f"Accuracy against position-derived labels: {accuracy:.4f}")
    print(f"Prediction counts: {json.dumps(pred_counts, sort_keys=True)}")
    print("Confusion matrix rows=true, cols=pred:")
    print(cm.tolist())

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pred_ids": pred_ids,
                "pred_labels": [classifier.id_to_label[int(pred_id)] for pred_id in pred_ids.tolist()],
                "probs": probs,
                "true_ids_from_positions": true_ids,
            },
            output_path,
        )
        print(f"Saved predictions: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a terrain classifier from concatenated long/short latent vectors."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a terrain classifier.")
    train_parser.add_argument("--latents-path", type=str, default=DEFAULT_LATENTS_PATH)
    train_parser.add_argument("--save-path", type=str, default="logs/terrain_latent_classifier.pt")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--hidden-dims", type=str, default="128,128")
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--no-stratify", action="store_true")
    train_parser.set_defaults(func=train_model)

    predict_parser = subparsers.add_parser("predict", help="Run a trained terrain classifier on a latents.pt file.")
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--latents-path", type=str, default=DEFAULT_LATENTS_PATH)
    predict_parser.add_argument("--output-path", type=str, default=None)
    predict_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(func=predict_from_file)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
