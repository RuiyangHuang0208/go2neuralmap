"""Path visualization utilities for navigation module."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from .planner import PlanResult


class PathVisualizer:
    """Visualization utility for cost maps and planned paths.

    This class provides methods to visualize:
    - Cost map heatmaps
    - Planned paths overlaid on cost maps
    - Start and goal positions
    """

    def __init__(self, figsize: Tuple[float, float] = (10, 8)):
        """Initialize the visualizer.

        Args:
            figsize: Figure size in inches (width, height). Default is (10, 8).
        """
        self.figsize = figsize

    def visualize_cost_map(
        self,
        cost_map: torch.Tensor,
        title: str = "Cost Map",
        show_colorbar: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Visualize a cost map as a heatmap.

        Args:
            cost_map: 2D cost map tensor of shape (H, W) or (H, W, 1).
            title: Plot title.
            show_colorbar: Whether to display a colorbar.
            ax: Optional matplotlib axes to draw on. Creates new figure if None.

        Returns:
            matplotlib Figure containing the visualization.
        """
        # Convert to numpy
        cost_map_np = self._to_numpy_2d(cost_map)

        # Create figure and axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot heatmap
        im = ax.imshow(cost_map_np, cmap="viridis", origin="upper", aspect="auto")

        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Cost")

        plt.tight_layout()
        return fig

    def visualize_path(
        self,
        cost_map: torch.Tensor,
        plan_result: PlanResult,
        title: str = "Planned Path",
        show_colorbar: bool = True,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Visualize a planned path overlaid on the cost map.

        Args:
            cost_map: 2D cost map tensor of shape (H, W) or (H, W, 1).
            plan_result: Planning result containing path and status.
            title: Plot title.
            show_colorbar: Whether to display a colorbar.
            ax: Optional matplotlib axes to draw on. Creates new figure if None.

        Returns:
            matplotlib Figure containing the visualization.
        """
        # Convert to numpy
        cost_map_np = self._to_numpy_2d(cost_map)

        # Create figure and axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Plot cost map heatmap
        im = ax.imshow(cost_map_np, cmap="viridis", origin="upper", aspect="auto")

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Cost")

        # Visualize path if successful
        if plan_result.path is not None:
            # Extract row and column coordinates
            path = plan_result.path
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]

            # Plot path line
            ax.plot(cols, rows, color="red", linewidth=2, label="Path")

            # Plot start position (green circle)
            if len(path) > 0:
                ax.plot(cols[0], rows[0], "go", markersize=10, label="Start")

            # Plot goal position (red star)
            if len(path) > 1:
                ax.plot(cols[-1], rows[-1], "y*", markersize=15, label="Goal")

            # Add legend
            ax.legend(loc="upper right")

        # Add status text
        status_text = f"Status: {plan_result.state.name}"
        if plan_result.state.name == "SUCCESS":
            status_text += f" | Cost: {plan_result.total_cost:.2f}"
        ax.text(
            0.02,
            0.98,
            status_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        plt.tight_layout()
        return fig

    def visualize_comparison(
        self,
        cost_map: torch.Tensor,
        plan_results: List[Tuple[PlanResult, str]],
        title: str = "Path Planning Comparison",
    ) -> plt.Figure:
        """Visualize multiple planning results for comparison.

        Args:
            cost_map: 2D cost map tensor.
            plan_results: List of (PlanResult, label) tuples.
            title: Overall figure title.

        Returns:
            matplotlib Figure with subplots for each planning result.
        """
        n_plots = len(plan_results)
        if n_plots == 0:
            raise ValueError("plan_results cannot be empty")

        fig, axes = plt.subplots(1, n_plots, figsize=(self.figsize[0] * n_plots, self.figsize[1]))
        if n_plots == 1:
            axes = [axes]

        for idx, (result, label) in enumerate(plan_results):
            self.visualize_path(
                cost_map,
                result,
                title=label,
                show_colorbar=(idx == 0),
                ax=axes[idx],
            )

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return fig

    def _to_numpy_2d(self, cost_map: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to 2D numpy array.

        Args:
            cost_map: Cost map tensor of shape (H, W) or (H, W, 1).

        Returns:
            2D numpy array of shape (H, W).
        """
        if cost_map.dim() not in (2, 3):
            raise ValueError(f"cost_map must be 2D or 3D, got {cost_map.dim()}D")

        if cost_map.dim() == 3:
            if cost_map.shape[2] != 1:
                raise ValueError(f"3D cost_map must have shape (H, W, 1), got {cost_map.shape}")
            cost_map_2d = cost_map.squeeze(-1)
        else:
            cost_map_2d = cost_map

        return cost_map_2d.detach().cpu().numpy()
