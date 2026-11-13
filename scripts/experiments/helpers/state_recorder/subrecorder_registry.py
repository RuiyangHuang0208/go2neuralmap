from .foot_print_plotter import FootPrint
from .cost_map_plotter import CostMap
from .trajectory_recorder import TrajectoryRecorder
from .cost_recorder import CostRecorder
from .latent_recorder import LatentRecorder
from .navigation_metrics_recorder import NavigationMetricsRecorder
from .uncertainty_footprint_recorder import UncertaintyFootprintRecorder

SubPlotterRegistry = {
    "foot_print": FootPrint,
    "cost_map": CostMap,
}

SubRecorderRegistry = {
    "trajectory_recorder": TrajectoryRecorder,
    "navigation_metrics_recorder": NavigationMetricsRecorder,
    "cost_recorder": CostRecorder,
    "latent_recorder": LatentRecorder,
    "uncertainty_footprint_recorder": UncertaintyFootprintRecorder,
}
