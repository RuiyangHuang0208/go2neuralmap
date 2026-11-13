import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

from .planner import BasePlanner, PlanResult, PlannerConfig, PlannerState

@dataclass
class AstarConfig(PlannerConfig):
    heuristic: str = "chebyshev_dist"



class AStarPlanner(BasePlanner):
    """A* path planner with configurable heuristic.

    Supported heuristics:
        - chebyshev_dist: max(abs(dx), abs(dy))
        - l2_dist: sqrt(dx^2 + dy^2)
        - null: 0.0

    The movement cost to an adjacent cell equals the destination cell's cost value.
    Diagonal movement cost is not scaled (same as orthogonal movement).
    """
    def __init__(self, cfg: AstarConfig=AstarConfig()):
        self.cfg = cfg

    @staticmethod
    def _chebyshev_dist(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return float(max(dx, dy))

    @staticmethod
    def _l2_dist(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        dx = pos[0] - goal[0]
        dy = pos[1] - goal[1]
        return float(math.sqrt(dx * dx + dy * dy))

    @staticmethod
    def _null_heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return 0.0

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: torch.Tensor,
    ) -> PlanResult:
        """Plan a path using A* algorithm.

        Args:
            start: Starting grid coordinate (row, col).
            goal: Goal grid coordinate (row, col).
            cost_map: 2D cost map tensor of shape (H, W) or (H, W, 1).

        Returns:
            PlanResult containing the path and total cost if successful.
        """
        # Validate inputs
        validation_result = self._validate_inputs(start, goal, cost_map)
        if validation_result is not None:
            return validation_result

        # Normalize cost map to 2D
        if cost_map.dim() == 3:
            cost_map_2d = cost_map.squeeze(-1)
        else:
            cost_map_2d = cost_map

        height, width = cost_map_2d.shape

        # Handle trivial case: start == goal
        if start == goal:
            return PlanResult(
                state=PlannerState.SUCCESS,
                path=[start],
                total_cost=cost_map_2d[start[0], start[1]].item(),
            )

        # Priority queue: (f_score, counter, position)
        # counter is used to break ties consistently
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0.0, counter, start))

        # Track visited nodes and their parents
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        # g_score includes the start cell cost
        start_cost = cost_map_2d[start[0], start[1]].item()
        g_score: dict[Tuple[int, int], float] = {start: start_cost}
        f_score: dict[Tuple[int, int], float] = {start: start_cost + self._heuristic(start, goal)}

        # Set of positions in open set for O(1) lookup
        open_set_hash = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            # Goal reached
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                total_cost = g_score[goal]
                return PlanResult(
                    state=PlannerState.SUCCESS,
                    path=path,
                    total_cost=total_cost,
                )

            # Explore neighbors
            for neighbor in self._get_neighbors(current, height, width):
                # Check if neighbor is an obstacle
                neighbor_cost = cost_map_2d[neighbor[0], neighbor[1]].item()
                if neighbor_cost >= self.cfg.obstacle_threshold:
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + neighbor_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return PlanResult(
            state=PlannerState.NO_PATH,
            path=None,
            total_cost=0.0,
        )

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Evaluate the configured heuristic.

        Args:
            pos: Current position.
            goal: Goal position.

        Returns:
            Heuristic estimate from pos to goal.
        """
        heuristic_registry: Dict[str, Callable[[Tuple[int, int], Tuple[int, int]], float]] = {
            "chebyshev_dist": self._chebyshev_dist,
            "l2_dist": self._l2_dist,
            "null": self._null_heuristic,
        }
        heuristic_name = getattr(self.cfg, "heuristic", "chebyshev_dist")
        heuristic_fn = heuristic_registry.get(heuristic_name)
        if heuristic_fn is None:
            available = ", ".join(sorted(heuristic_registry))
            raise ValueError(
                f"Unknown A* heuristic '{heuristic_name}'. Available heuristics: {available}"
            )
        return heuristic_fn(pos, goal)

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to goal.

        Args:
            came_from: Dictionary mapping each node to its parent.
            current: Goal position.

        Returns:
            List of coordinates from start to goal (inclusive).
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
