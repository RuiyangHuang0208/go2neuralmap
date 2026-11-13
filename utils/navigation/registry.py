from .astar import AstarConfig, AStarPlanner
from .edge_cost_astar import EdgeCostAstarConfig, EdgeCostAStarPlanner

PlannerRegistry = {
    "astar": (AStarPlanner, AstarConfig()),
    "edge_cost_astar": (EdgeCostAStarPlanner, EdgeCostAstarConfig()),
}
