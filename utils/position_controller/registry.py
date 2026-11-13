from .position_controller_bezier import PositionControllerBezier, PositionControllerBezierConfig
from .position_controller_polynomial_time import (
    PositionControllerPolynomialTime,
    PositionControllerPolynomialTimeConfig,
)
from .position_controller_rotation_first import PositionControllerRotationFirst, PositionControllerRotationFirstConfig

PositionControllerRegistry = {
    "position_controller_rotation_first": (PositionControllerRotationFirst, PositionControllerRotationFirstConfig()),
    "position_controller_polynomial_time": (PositionControllerPolynomialTime, PositionControllerPolynomialTimeConfig()),
    "position_controller_bezier": (PositionControllerBezier, PositionControllerBezierConfig()),
}
