#!/usr/bin/env python3
"""Script to plan and visualize a path on a loaded cost map.

This script loads a cost map from a .pt file, plans a path using A*,
and visualizes the result.

Example usage:
    # Plan from (0, 0) to (50, 50) with default obstacle threshold
    python scripts/plan_path.py cost_map.pt --start 0 0 --goal 50 50

    # Plan with custom obstacle threshold
    python scripts/plan_path.py cost_map.pt --start 0 0 --goal 50 50 --obstacle-threshold 300

    # Save visualization to file
    python scripts/plan_path.py cost_map.pt --start 0 0 --goal 50 50 --output path_result.png
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
from time import time

from utils.navigation import AStarPlanner, PlannerConfig, PathVisualizer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plan and visualize a path on a cost map loaded from .pt file"
    )
    parser.add_argument(
        "cost_map_path",
        type=str,
        help="Path to the cost map .pt file",
    )
    parser.add_argument(
        "--start",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("ROW", "COL"),
        help="Start position (row, col). Default: (0, 0)",
    )
    parser.add_argument(
        "--goal",
        type=int,
        nargs=2,
        default=None,
        metavar=("ROW", "COL"),
        help="Goal position (row, col). Default: opposite corner of map",
    )
    parser.add_argument(
        "--obstacle-threshold",
        type=float,
        default=500.0,
        help="Cost threshold for obstacles. Cells with cost >= this are blocked. Default: 500.0",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path to save the visualization (e.g., path_result.png). Shows interactive window if not set.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed planning information",
    )
    return parser.parse_args()


def load_cost_map(path: str) -> torch.Tensor:
    """Load cost map from .pt file.

    Args:
        path: Path to the .pt file.

    Returns:
        2D cost map tensor.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded tensor has invalid shape.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cost map file not found: {path}")

    cost_map = torch.load(path, map_location="cpu", weights_only=False)

    # Handle batched cost maps (num_envs, H, W, 1) -> extract first env
    if cost_map.dim() == 4:
        print(f"Warning: Loaded cost map has 4D shape {cost_map.shape}, extracting first environment")
        cost_map = cost_map[0].squeeze(-1)
    elif cost_map.dim() == 3:
        if cost_map.shape[2] == 1:
            cost_map = cost_map.squeeze(-1)
        else:
            # Assume it's (H, W, 1) single env
            pass
    elif cost_map.dim() != 2:
        raise ValueError(f"Expected 2D cost map, got shape {cost_map.shape}")

    return cost_map


def plan_path(
    cost_map: torch.Tensor,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    obstacle_threshold: float,
) -> Tuple[bool, list, float]:
    """Plan a path on the cost map.

    Args:
        cost_map: 2D cost map tensor.
        start: Start position (row, col).
        goal: Goal position (row, col).
        obstacle_threshold: Cost threshold for obstacles.

    Returns:
        Tuple of (success, path, total_cost).
    """
    cfg = PlannerConfig(obstacle_threshold=obstacle_threshold)
    planner = AStarPlanner(cfg)

    result = planner.plan(start, goal, cost_map)

    return result.state.name == "SUCCESS", result.path, result.total_cost


def main():
    """Main entry point."""
    args = parse_args()

    # Load cost map
    print(f"Loading cost map from: {args.cost_map_path}")
    try:
        cost_map = load_cost_map(args.cost_map_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading cost map: {e}")
        return 1

    valid_cost_map = cost_map[4:-4, 4:-4]
    cost_map = valid_cost_map

    print(f"Cost map shape: {cost_map.shape}")
    print(f"Cost range: [{cost_map.min().item():.2f}, {cost_map.max().item():.2f}]")

    # Determine goal position
    height, width = cost_map.shape
    if args.goal is None:
        goal = (height - 1, width - 1)
    else:
        goal = tuple(args.goal)

    start = tuple(args.start)

    print(f"Planning path from {start} to {goal}")
    print(f"Obstacle threshold: {args.obstacle_threshold}")

    # Plan path
    t_start = time()
    success, path, total_cost = plan_path(
        cost_map,
        start,
        goal,
        args.obstacle_threshold,
    )
    t_end = time()

    if not success:
        print("Path planning failed!")
        # Try to determine failure reason
        from utils.navigation import AStarPlanner, PlannerConfig, PlannerState

        cfg = PlannerConfig(obstacle_threshold=args.obstacle_threshold)
        planner = AStarPlanner(cfg)
        result = planner.plan(start, goal, cost_map)
        print(f"Failure reason: {result.state.name}")
        return 1

    print(f"Path found!")
    print(f"Path length: {len(path)} waypoints")
    print(f"Total cost: {total_cost:.2f}")
    print(f"Planning time: {t_end - t_start:.2f}s")

    if args.verbose:
        print(f"Path: {path}")

    # Visualize
    from utils.navigation import PathVisualizer, PlanResult, PlannerState

    visualizer = PathVisualizer()
    plan_result = PlanResult(
        state=PlannerState.SUCCESS,
        path=path,
        total_cost=total_cost,
    )
    fig = visualizer.visualize_path(
        cost_map,
        plan_result,
        title=f"Planned Path (Cost: {total_cost:.2f})",
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_path}")
    else:
        print("Displaying visualization window... (close window to exit)")
        import matplotlib.pyplot as plt
        plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
