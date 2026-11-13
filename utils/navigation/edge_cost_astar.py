import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .planner import BasePlanner, PlanResult, PlannerConfig, PlannerState


@dataclass
class EdgeCostAstarConfig(PlannerConfig):
    heuristic: str = "manhattan_dist"
    heuristic_min_step_cost: float = 0.0


class EdgeCostAStarPlanner(BasePlanner):
    """A* planner for 4-neighbor directed edge costs stored as (H, W, 4)."""

    POS_X = 0
    NEG_X = 1
    POS_Y = 2
    NEG_Y = 3

    def __init__(self, cfg: EdgeCostAstarConfig = EdgeCostAstarConfig()):
        self.cfg = cfg

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        cost_map: torch.Tensor,
    ) -> PlanResult:
        if cost_map.dim() != 3 or cost_map.shape[2] != 4:
            return PlanResult(
                state=PlannerState.INVALID_MAP,
                path=None,
                total_cost=0.0,
            )

        height, width, _ = cost_map.shape
        point_validation = self._validate_grid_points(start, goal, height, width)
        if point_validation is not None:
            return point_validation

        if start == goal:
            return PlanResult(
                state=PlannerState.SUCCESS,
                path=[start],
                total_cost=0.0,
            )

        counter = 0
        open_set = []
        heapq.heappush(open_set, (0.0, counter, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, goal)}
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            if current_f > f_score.get(current, float("inf")):
                continue

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return PlanResult(
                    state=PlannerState.SUCCESS,
                    path=path,
                    total_cost=g_score[goal],
                )

            for neighbor in self._get_cardinal_neighbors(current, height, width):
                edge_cost = self._get_edge_cost(cost_map, current, neighbor)
                if edge_cost >= self.cfg.obstacle_threshold:
                    continue

                tentative_g = g_score[current] + edge_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        return PlanResult(
            state=PlannerState.NO_PATH,
            path=None,
            total_cost=0.0,
        )

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        if self.cfg.heuristic == "manhattan_dist":
            dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            return float(dist * self.cfg.heuristic_min_step_cost)
        if self.cfg.heuristic == "l2_dist":
            dx = pos[0] - goal[0]
            dy = pos[1] - goal[1]
            dist = math.sqrt(dx * dx + dy * dy)
            return float(dist * self.cfg.heuristic_min_step_cost)
        raise ValueError(
            f"Unknown edge-cost A* heuristic '{self.cfg.heuristic}'. Available heuristics: l2_dist, manhattan_dist"
        )

    def _get_edge_cost(
        self,
        cost_map: torch.Tensor,
        current: Tuple[int, int],
        neighbor: Tuple[int, int],
    ) -> float:
        channel = self._direction_to_channel(neighbor[0] - current[0], neighbor[1] - current[1])
        return float(cost_map[current[0], current[1], channel].item())

    def _direction_to_channel(self, delta_x: int, delta_y: int) -> int:
        if delta_x == 1 and delta_y == 0:
            return self.POS_X
        if delta_x == -1 and delta_y == 0:
            return self.NEG_X
        if delta_x == 0 and delta_y == 1:
            return self.POS_Y
        if delta_x == 0 and delta_y == -1:
            return self.NEG_Y
        raise ValueError(f"Expected a 4-neighbor move, got delta=({delta_x}, {delta_y})")

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
