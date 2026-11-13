from .locomotion_base_model import LocoMotionBaseModel, LocoMotionBaseModelCfg
from .locomotion_model import LocoMotionModel, LocoMotionModelCfg

LocoMotionModelRegistry = {
    "locomotion_model": (LocoMotionModel, LocoMotionModelCfg())
}
