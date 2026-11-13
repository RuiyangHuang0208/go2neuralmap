from .position_navigation import PositionNavigation, PositionNavigationConfig
from .point_to_point_navigation import PointToPointNavigation, PointToPointNavigationConfig
from .keyboard_input import KeyboardInput, KeyboardInputConfig
from .diffusion_policy import (
    DiffusionPolicyVelCommandGenerator,
    DiffusionPolicyVelCommandGeneratorConfig,
)
from .residual_diffusion_policy import (
    ResidualDiffusionPolicyVelCommandGenerator,
    ResidualDiffusionPolicyVelCommandGeneratorConfig,
)
from .rl_nav_policy import (
    RLNavPolicyVelCommandGenerator,
    RLNavPolicyVelCommandGeneratorConfig,
)
from .vel_command_generator import VelCommandGenerator, VelCommandGeneratorConfig

VelCommandGeneratorRegistry = {
    "position_navigation": (PositionNavigation, PositionNavigationConfig()),
    "point_to_point_navigation": (PointToPointNavigation, PointToPointNavigationConfig()),
    "keyboard_input": (KeyboardInput, KeyboardInputConfig()),
    "diffusion_policy": (
        DiffusionPolicyVelCommandGenerator,
        DiffusionPolicyVelCommandGeneratorConfig(),
    ),
    "residual_diffusion_policy": (
        ResidualDiffusionPolicyVelCommandGenerator,
        ResidualDiffusionPolicyVelCommandGeneratorConfig(),
    ),
    "rl_nav_policy": (
        RLNavPolicyVelCommandGenerator,
        RLNavPolicyVelCommandGeneratorConfig(),
    ),
    "null_impl": (VelCommandGenerator, VelCommandGeneratorConfig()),
}
