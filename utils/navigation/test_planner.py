"""Test cases for navigation module path planners."""

import pytest
import torch

from utils.navigation import (
    AStarPlanner,
    EdgeCostAStarPlanner,
    PlanResult,
    PlannerConfig,
    PlannerState,
)
from utils.navigation.astar import AstarConfig
from utils.navigation.edge_cost_astar import EdgeCostAstarConfig
from utils.navigation.planner import BasePlanner


class TestPlannerConfig:
    """Test cases for PlannerConfig."""

    def test_default_obstacle_threshold(self):
        """Test default obstacle threshold value."""
        cfg = PlannerConfig()
        assert cfg.obstacle_threshold == 500.0

    def test_custom_obstacle_threshold(self):
        """Test custom obstacle threshold value."""
        cfg = PlannerConfig(obstacle_threshold=100.0)
        assert cfg.obstacle_threshold == 100.0


class TestPlanResult:
    """Test cases for PlanResult dataclass."""

    def test_success_result(self):
        """Test successful planning result."""
        result = PlanResult(
            state=PlannerState.SUCCESS,
            path=[(0, 0), (1, 1), (2, 2)],
            total_cost=15.0,
        )
        assert result.state == PlannerState.SUCCESS
        assert len(result.path) == 3
        assert result.total_cost == 15.0

    def test_failure_result(self):
        """Test failed planning result."""
        result = PlanResult(
            state=PlannerState.NO_PATH,
            path=None,
            total_cost=0.0,
        )
        assert result.state == PlannerState.NO_PATH
        assert result.path is None
        assert result.total_cost == 0.0


class TestBasePlanner:
    """Test cases for BasePlanner abstract class."""

    def test_base_planner_is_abstract(self):
        """Test that BasePlanner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePlanner()  # type: ignore[abstract]


class TestAStarPlanner:
    """Test cases for AStarPlanner."""

    @pytest.fixture
    def planner(self):
        """Create a default AStarPlanner for testing."""
        return AStarPlanner()

    @pytest.fixture
    def simple_cost_map(self):
        """Create a simple uniform cost map for testing."""
        return torch.ones(10, 10) * 10.0

    def test_plan_trivial_path(self, planner: AStarPlanner):
        """Test planning when start equals goal."""
        cost_map = torch.ones(10, 10) * 5.0
        result = planner.plan((5, 5), (5, 5), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path == [(5, 5)]
        assert result.total_cost == 5.0

    def test_plan_straight_path(self, planner: AStarPlanner):
        """Test planning a straight horizontal path."""
        # Create a corridor to force horizontal path
        cost_map = torch.ones(10, 10) * 100.0
        cost_map[5, :] = 10.0  # Clear row 5
        
        result = planner.plan((5, 0), (5, 5), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None
        assert len(result.path) == 6  # (5,0) to (5,5) inclusive
        assert result.path[0] == (5, 0)
        assert result.path[-1] == (5, 5)
        # Total cost = 6 cells * 10.0 (including start)
        assert result.total_cost == 60.0

    def test_plan_diagonal_path(self, planner: AStarPlanner):
        """Test planning a diagonal path."""
        cost_map = torch.ones(10, 10) * 10.0
        result = planner.plan((0, 0), (5, 5), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None
        # Chebyshev distance = 5, so optimal path has 6 points
        assert len(result.path) == 6
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (5, 5)

    def test_default_heuristic_is_chebyshev(self):
        """Test that default A* config keeps the existing Chebyshev heuristic."""
        planner = AStarPlanner()
        assert planner.cfg.heuristic == "chebyshev_dist"
        assert planner._heuristic((0, 0), (3, 4)) == 4.0

    def test_l2_heuristic(self):
        """Test Euclidean heuristic value."""
        planner = AStarPlanner(AstarConfig(heuristic="l2_dist"))
        assert planner._heuristic((0, 0), (3, 4)) == 5.0

    def test_null_heuristic(self):
        """Test null heuristic value."""
        planner = AStarPlanner(AstarConfig(heuristic="null"))
        assert planner._heuristic((0, 0), (3, 4)) == 0.0

    def test_invalid_heuristic_name(self):
        """Test invalid heuristic names fail explicitly."""
        planner = AStarPlanner(AstarConfig(heuristic="invalid"))
        with pytest.raises(ValueError, match="Unknown A\\* heuristic"):
            planner._heuristic((0, 0), (1, 1))

    def test_invalid_start_out_of_bounds(self, planner: AStarPlanner):
        """Test planning with start position out of bounds."""
        cost_map = torch.ones(10, 10)
        result = planner.plan((10, 10), (5, 5), cost_map)

        assert result.state == PlannerState.INVALID_START
        assert result.path is None
        assert result.total_cost == 0.0

    def test_invalid_goal_out_of_bounds(self, planner: AStarPlanner):
        """Test planning with goal position out of bounds."""
        cost_map = torch.ones(10, 10)
        result = planner.plan((5, 5), (-1, -1), cost_map)

        assert result.state == PlannerState.INVALID_GOAL
        assert result.path is None
        assert result.total_cost == 0.0

    def test_invalid_start_obstacle(self, planner: AStarPlanner):
        """Test planning when start is an obstacle."""
        cost_map = torch.ones(10, 10) * 10.0
        cost_map[0, 0] = 600.0  # Obstacle

        result = planner.plan((0, 0), (5, 5), cost_map)

        assert result.state == PlannerState.INVALID_START
        assert result.path is None
        assert result.total_cost == 0.0

    def test_invalid_goal_obstacle(self, planner: AStarPlanner):
        """Test planning when goal is an obstacle."""
        cost_map = torch.ones(10, 10) * 10.0
        cost_map[5, 5] = 600.0  # Obstacle

        result = planner.plan((0, 0), (5, 5), cost_map)

        assert result.state == PlannerState.INVALID_GOAL
        assert result.path is None
        assert result.total_cost == 0.0

    def test_no_path_blocked(self, planner: AStarPlanner):
        """Test planning when path is completely blocked."""
        cost_map = torch.ones(10, 10) * 10.0
        # Create a wall of obstacles
        cost_map[:, 5] = 600.0

        result = planner.plan((0, 0), (9, 9), cost_map)

        assert result.state == PlannerState.NO_PATH
        assert result.path is None
        assert result.total_cost == 0.0

    def test_invalid_map_3d_wrong_shape(self, planner: AStarPlanner):
        """Test planning with invalid 3D cost map shape."""
        cost_map = torch.ones(10, 10, 3)  # Invalid: last dim should be 1

        result = planner.plan((0, 0), (5, 5), cost_map)

        assert result.state == PlannerState.INVALID_MAP
        assert result.path is None
        assert result.total_cost == 0.0

    def test_3d_cost_map_with_single_channel(self, planner: AStarPlanner):
        """Test planning with valid 3D cost map (H, W, 1)."""
        cost_map = torch.ones(10, 10, 1) * 10.0
        result = planner.plan((0, 0), (5, 5), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None

    def test_custom_obstacle_threshold(self):
        """Test planning with custom obstacle threshold."""
        cfg = PlannerConfig(obstacle_threshold=50.0)
        planner = AStarPlanner(cfg)

        cost_map = torch.ones(10, 10) * 40.0  # Below threshold
        cost_map[5, 5] = 60.0  # Above threshold

        # Goal is obstacle with custom threshold
        result = planner.plan((0, 0), (5, 5), cost_map)
        assert result.state == PlannerState.INVALID_GOAL

        # Path around obstacle should succeed
        result = planner.plan((0, 0), (0, 5), cost_map)
        assert result.state == PlannerState.SUCCESS

    def test_path_cost_calculation(self, planner: AStarPlanner):
        """Test that path cost is calculated correctly."""
        # Create a corridor where only one path exists
        cost_map = torch.ones(10, 10) * 100.0  # High cost background
        cost_map[0, :] = 0.0  # Clear first row
        
        cost_map[0, 0] = 5.0
        cost_map[0, 1] = 10.0
        cost_map[0, 2] = 15.0

        result = planner.plan((0, 0), (0, 2), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path == [(0, 0), (0, 1), (0, 2)]
        # Cost = sum of all visited cell costs (including start and goal)
        assert result.total_cost == 5.0 + 10.0 + 15.0

    def test_8_direction_movement(self, planner: AStarPlanner):
        """Test that 8-directional movement is supported."""
        cost_map = torch.ones(10, 10) * 10.0

        # Diagonal movement should be possible
        result = planner.plan((0, 0), (3, 3), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None
        # Path should use diagonal moves (length = 4 for Chebyshev distance 3)
        assert len(result.path) == 4

    def test_avoid_high_cost_cells(self):
        """Test that planner avoids high-cost cells when possible."""
        cost_map = torch.ones(10, 10) * 10.0
        # Create a high-cost region in the middle
        cost_map[4:7, 4:7] = 100.0

        planner_low_threshold = AStarPlanner(PlannerConfig(obstacle_threshold=500.0))
        result = planner_low_threshold.plan((0, 0), (9, 9), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None
        # Path should exist and may go around or through high-cost region


class TestAStarPlannerBatchInterface:
    """Test cases for future batch planning interface."""

    def test_single_env_from_batch(self):
        """Test extracting single environment from batched cost map."""
        planner = AStarPlanner()

        # Simulate batched cost map (num_envs=4, H=10, W=10)
        batched_cost_map = torch.ones(4, 10, 10) * 10.0

        # Extract single environment (this is the interface reservation test)
        single_env_map = batched_cost_map[0]  # Shape: (10, 10)

        result = planner.plan((0, 0), (5, 5), single_env_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None


class TestEdgeCostAStarPlanner:
    @pytest.fixture
    def planner(self):
        return EdgeCostAStarPlanner()

    def test_plan_trivial_path(self, planner: EdgeCostAStarPlanner):
        cost_map = torch.ones(4, 4, 4) * 2.0
        result = planner.plan((1, 1), (1, 1), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path == [(1, 1)]
        assert result.total_cost == 0.0

    def test_prefers_lower_edge_cost_path(self, planner: EdgeCostAStarPlanner):
        cost_map = torch.ones(3, 3, 4) * 100.0

        cost_map[0, 0, 2] = 1.0
        cost_map[0, 1, 2] = 1.0
        cost_map[0, 2, 0] = 1.0
        cost_map[1, 2, 0] = 1.0

        cost_map[0, 0, 0] = 50.0
        cost_map[1, 0, 0] = 50.0
        cost_map[2, 0, 2] = 50.0
        cost_map[2, 1, 2] = 50.0

        result = planner.plan((0, 0), (2, 2), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        assert result.total_cost == pytest.approx(4.0)

    def test_blocked_edges_produce_no_path(self, planner: EdgeCostAStarPlanner):
        cost_map = torch.zeros(2, 2, 4)
        cost_map[0, 0, 0] = 600.0
        cost_map[0, 0, 2] = 600.0

        result = planner.plan((0, 0), (1, 1), cost_map)

        assert result.state == PlannerState.NO_PATH
        assert result.path is None

    def test_invalid_map_shape(self, planner: EdgeCostAStarPlanner):
        result = planner.plan((0, 0), (1, 1), torch.ones(4, 4))
        assert result.state == PlannerState.INVALID_MAP

    def test_invalid_heuristic_name(self):
        planner = EdgeCostAStarPlanner(EdgeCostAstarConfig(heuristic="bad"))
        with pytest.raises(ValueError, match="Unknown edge-cost A\\* heuristic"):
            planner.plan((0, 0), (1, 1), torch.zeros(2, 2, 4))

    def test_l2_heuristic_value(self):
        planner = EdgeCostAStarPlanner(
            EdgeCostAstarConfig(heuristic="l2_dist", heuristic_min_step_cost=2.0)
        )
        assert planner._heuristic((0, 0), (3, 4)) == pytest.approx(10.0)

    def test_l2_heuristic_planning(self):
        planner = EdgeCostAStarPlanner(
            EdgeCostAstarConfig(heuristic="l2_dist", heuristic_min_step_cost=0.0)
        )
        cost_map = torch.ones(3, 3, 4)

        result = planner.plan((0, 0), (2, 2), cost_map)

        assert result.state == PlannerState.SUCCESS
        assert result.path is not None
