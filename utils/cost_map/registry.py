from .ema_cost_map import EMACostMap, EMACostMapCfg
from .transition_cost_map import TransitionCostMap, TransitionCostMapCfg
from .directional_ema_transition_cost_map import (
    DirectionalEMATransitionCostMap,
    DirectionalEMATransitionCostMapCfg,
)
from .directional_ema_velocity_cost_map import (
    DirectionalEMAVelocityCostMap,
    DirectionalEMAVelocityCostMapCfg,
)
from .collision_directional_ema_velocity_cost_map import (
    CollisionDirectionalEMAVelocityCostMap,
    CollisionDirectionalEMAVelocityCostMapCfg,
)

CostMapRegistry = {
    "ema_cost_map": (EMACostMap, EMACostMapCfg()),
    "transition_cost_map": (TransitionCostMap, TransitionCostMapCfg()),
    "directional_ema_transition_cost_map": (
        DirectionalEMATransitionCostMap,
        DirectionalEMATransitionCostMapCfg(),
    ),
    "directional_ema_velocity_cost_map": (
        DirectionalEMAVelocityCostMap,
        DirectionalEMAVelocityCostMapCfg(),
    ),
    "collision_directional_ema_velocity_cost_map": (
        CollisionDirectionalEMAVelocityCostMap,
        CollisionDirectionalEMAVelocityCostMapCfg(),
    ),
}
