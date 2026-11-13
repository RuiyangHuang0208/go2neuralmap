from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


DEFAULT_DATASET_DIR = (
    "logs/2026-05-18_d00496e/demo_plane_real_path_dataset_5env_1_14-51-20/dataset"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether near-goal velocity command labels actually reach or leave the goal."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--near-threshold", type=float, default=1.0, help="Meters from start_xy to goal_xy.")
    parser.add_argument("--goal-tolerance", type=float, default=0.35, help="Meters counted as reaching the goal.")
    parser.add_argument("--leave-tolerance", type=float, default=0.5, help="Meters counted as leaving after reaching.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plot-dir", default=None, help="Optional directory for worst-case near-goal plots.")
    parser.add_argument("--num-plots", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser()
    metadata = load_metadata(dataset_dir)
    dt = float(metadata.get("collector", {}).get("velocity_command_dt", metadata.get("velocity_command_dt", 0.5)))
    least_count = float(metadata.get("least_count", 0.25))

    rows = []
    total = 0
    for shard_path in tqdm(sorted(dataset_dir.glob("shard_*.pt"))):
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        count = int(shard["start_xy"].shape[0])
        for local_idx in range(count):
            if args.max_samples is not None and total >= int(args.max_samples):
                break
            row = analyze_sample(
                shard=shard,
                shard_name=shard_path.name,
                local_idx=local_idx,
                dt=dt,
                goal_tolerance=float(args.goal_tolerance),
                leave_tolerance=float(args.leave_tolerance),
            )
            rows.append(row)
            total += 1
        if args.max_samples is not None and total >= int(args.max_samples):
            break

    near_rows = [row for row in rows if row["start_goal_dist"] <= float(args.near_threshold)]
    print_summary(rows, near_rows, args)

    if args.plot_dir is not None:
        plot_dir = Path(args.plot_dir).expanduser()
        plot_dir.mkdir(parents=True, exist_ok=True)
        worst = sorted(
            near_rows,
            key=lambda row: (row["reached_goal"], row["final_goal_dist"], -row["start_goal_dist"]),
            reverse=True,
        )[: int(args.num_plots)]
        for idx, row in enumerate(worst):
            plot_path = plot_dir / f"near_goal_{idx:03d}_{row['shard_name']}_{row['local_idx']}.png"
            plot_case(row, plot_path, least_count=least_count)
        print(f"Saved {len(worst)} plots to {plot_dir}")


def load_metadata(dataset_dir: Path) -> Dict:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def analyze_sample(
    *,
    shard: Dict[str, torch.Tensor],
    shard_name: str,
    local_idx: int,
    dt: float,
    goal_tolerance: float,
    leave_tolerance: float,
) -> Dict:
    start_xy = shard["start_xy"][local_idx].float()
    goal_xy = shard["goal_xy"][local_idx].float()
    commands = shard["velocity_commands"][local_idx].float()
    mask = shard["velocity_command_mask"][local_idx].bool()
    valid_commands = commands[mask]
    path = integrate_commands(start_xy, valid_commands, dt=dt)
    dists = torch.norm(path - goal_xy.view(1, 2), dim=-1)

    reached_indices = torch.nonzero(dists <= goal_tolerance, as_tuple=False).flatten()
    reached_goal = bool(reached_indices.numel() > 0)
    first_reach_idx = int(reached_indices[0].item()) if reached_goal else -1
    left_after_reach = False
    if reached_goal and first_reach_idx + 1 < int(dists.numel()):
        left_after_reach = bool((dists[first_reach_idx + 1 :] > leave_tolerance).any().item())

    row = {
        "shard_name": shard_name,
        "local_idx": int(local_idx),
        "start_xy": start_xy,
        "goal_xy": goal_xy,
        "path": path,
        "dists": dists,
        "valid_horizon": int(mask.sum().item()),
        "start_goal_dist": float(torch.norm(goal_xy - start_xy).item()),
        "min_goal_dist": float(dists.min().item()) if dists.numel() > 0 else math.inf,
        "final_goal_dist": float(dists[-1].item()) if dists.numel() > 0 else math.inf,
        "reached_goal": reached_goal,
        "first_reach_idx": first_reach_idx,
        "left_after_reach": left_after_reach,
        "done": bool(shard["done"][local_idx].item()) if "done" in shard else False,
        "goal_from_context": bool(shard["goal_from_context"][local_idx].item()) if "goal_from_context" in shard else False,
    }
    if "step" in shard:
        row["step"] = int(shard["step"][local_idx].item())
    if "env_id" in shard:
        row["env_id"] = int(shard["env_id"][local_idx].item())
    return row


def integrate_commands(start_xy: torch.Tensor, commands: torch.Tensor, *, dt: float) -> torch.Tensor:
    pos = start_xy.float().clone()
    points = [pos.clone()]
    for command in commands:
        vx = float(command[0].item())
        vy = float(command[1].item())
        heading = float(command[2].item())
        if not math.isfinite(heading):
            heading = 0.0
        dx = (math.cos(heading) * vx - math.sin(heading) * vy) * float(dt)
        dy = (math.sin(heading) * vx + math.cos(heading) * vy) * float(dt)
        pos = pos + torch.tensor([dx, dy], dtype=torch.float32)
        points.append(pos.clone())
    return torch.stack(points)


def print_summary(rows: List[Dict], near_rows: List[Dict], args: argparse.Namespace) -> None:
    print(f"total_samples={len(rows)}")
    print(
        f"near_goal_samples={len(near_rows)} "
        f"threshold={float(args.near_threshold):.3f}m "
        f"ratio={safe_ratio(len(near_rows), len(rows)):.4f}"
    )
    summarize_group("all", rows, args)
    summarize_group("near_goal", near_rows, args)
    print_distance_bins(rows, args)

    if near_rows:
        near_not_reached = [row for row in near_rows if not row["reached_goal"]]
        near_reached_left = [row for row in near_rows if row["left_after_reach"]]
        final_far = [row for row in near_rows if row["final_goal_dist"] > float(args.leave_tolerance)]
        print(f"near_not_reached={len(near_not_reached)} ratio={safe_ratio(len(near_not_reached), len(near_rows)):.4f}")
        print(f"near_reached_then_left={len(near_reached_left)} ratio={safe_ratio(len(near_reached_left), len(near_rows)):.4f}")
        print(f"near_final_far>{float(args.leave_tolerance):.2f}m={len(final_far)} ratio={safe_ratio(len(final_far), len(near_rows)):.4f}")
        print_examples("worst_near_final_distance", sorted(near_rows, key=lambda r: r["final_goal_dist"], reverse=True)[:10])
        print_examples("near_reached_then_left_examples", near_reached_left[:10])


def summarize_group(name: str, rows: List[Dict], args: argparse.Namespace) -> None:
    if not rows:
        print(f"{name}: empty")
        return
    start = torch.tensor([row["start_goal_dist"] for row in rows])
    min_dist = torch.tensor([row["min_goal_dist"] for row in rows])
    final = torch.tensor([row["final_goal_dist"] for row in rows])
    valid_h = torch.tensor([row["valid_horizon"] for row in rows], dtype=torch.float32)
    reached = sum(int(row["reached_goal"]) for row in rows)
    left = sum(int(row["left_after_reach"]) for row in rows)
    done = sum(int(row["done"]) for row in rows)
    context_goal = sum(int(row["goal_from_context"]) for row in rows)
    print(f"{name}: count={len(rows)}")
    print(f"  start_goal_dist: {describe(start)}")
    print(f"  min_goal_dist:   {describe(min_dist)}")
    print(f"  final_goal_dist: {describe(final)}")
    print(f"  valid_horizon:   {describe(valid_h)}")
    print(f"  reached_goal<={float(args.goal_tolerance):.2f}m: {reached} ratio={safe_ratio(reached, len(rows)):.4f}")
    print(f"  left_after_reach>{float(args.leave_tolerance):.2f}m: {left} ratio={safe_ratio(left, len(rows)):.4f}")
    print(f"  done_anchor: {done} ratio={safe_ratio(done, len(rows)):.4f}")
    print(f"  goal_from_context: {context_goal} ratio={safe_ratio(context_goal, len(rows)):.4f}")


def describe(values: torch.Tensor) -> str:
    values = values.float()
    if values.numel() == 0:
        return "empty"
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]))
    return (
        f"mean={values.mean().item():.4f} "
        f"q0={quantiles[0].item():.4f} q25={quantiles[1].item():.4f} "
        f"q50={quantiles[2].item():.4f} q75={quantiles[3].item():.4f} "
        f"q90={quantiles[4].item():.4f} q99={quantiles[5].item():.4f} "
        f"max={quantiles[6].item():.4f}"
    )


def print_examples(name: str, rows: List[Dict]) -> None:
    print(f"{name}:")
    if not rows:
        print("  none")
        return
    for row in rows:
        print(
            "  "
            f"{row['shard_name']}[{row['local_idx']}] "
            f"env={row.get('env_id', '?')} step={row.get('step', '?')} "
            f"start={row['start_goal_dist']:.3f} min={row['min_goal_dist']:.3f} "
            f"final={row['final_goal_dist']:.3f} reached={row['reached_goal']} "
            f"left={row['left_after_reach']} horizon={row['valid_horizon']} "
            f"done={row['done']} context_goal={row['goal_from_context']}"
        )


def print_distance_bins(rows: List[Dict], args: argparse.Namespace) -> None:
    if not rows:
        return
    bins = [0.0, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, math.inf]
    print("distance_bins:")
    for low, high in zip(bins[:-1], bins[1:]):
        group = [
            row
            for row in rows
            if row["start_goal_dist"] >= low and row["start_goal_dist"] < high
        ]
        if not group:
            continue
        final = torch.tensor([row["final_goal_dist"] for row in group])
        min_dist = torch.tensor([row["min_goal_dist"] for row in group])
        valid_h = torch.tensor([row["valid_horizon"] for row in group], dtype=torch.float32)
        reached = sum(int(row["reached_goal"]) for row in group)
        left = sum(int(row["left_after_reach"]) for row in group)
        label = f"[{low:.2f}, {'inf' if math.isinf(high) else f'{high:.2f}'})"
        print(
            "  "
            f"{label} count={len(group)} "
            f"valid_h_mean={valid_h.mean().item():.2f} valid_h_q50={torch.quantile(valid_h, 0.5).item():.1f} "
            f"min_q50={torch.quantile(min_dist, 0.5).item():.3f} "
            f"final_q50={torch.quantile(final, 0.5).item():.3f} final_q90={torch.quantile(final, 0.9).item():.3f} "
            f"reached={safe_ratio(reached, len(group)):.3f} left={safe_ratio(left, len(group)):.3f}"
        )


def plot_case(row: Dict, path: Path, *, least_count: float) -> None:
    xy = row["path"]
    goal = row["goal_xy"]
    start = row["start_xy"]
    dists = row["dists"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(xy[:, 1], xy[:, 0], marker="o", color="tab:blue", label="integrated label")
    ax.scatter(start[1], start[0], color="black", s=45, label="start", zorder=3)
    ax.scatter(goal[1], goal[0], color="lime", edgecolors="black", s=60, label="goal", zorder=3)
    tol_circle = plt.Circle((goal[1].item(), goal[0].item()), 0.35, fill=False, color="tab:green", linestyle="--")
    ax.add_patch(tol_circle)
    ax.set_aspect("equal")
    ax.set_title(
        f"{row['shard_name']}[{row['local_idx']}]\n"
        f"start={row['start_goal_dist']:.2f} min={row['min_goal_dist']:.2f} final={row['final_goal_dist']:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(torch.arange(dists.numel()), dists, marker="o")
    ax.axhline(0.35, color="tab:green", linestyle="--", label="goal tolerance")
    ax.set_xlabel("integrated path step")
    ax.set_ylabel("distance to goal (m)")
    ax.set_title(f"least_count={least_count}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def safe_ratio(num: int, denom: int) -> float:
    return float(num) / float(denom) if denom else 0.0


if __name__ == "__main__":
    main()
