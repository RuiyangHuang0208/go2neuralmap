from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CostEstimatorConfig:
    pass


class CostEstimator(ABC):
    def __init__(self, cfg: CostEstimatorConfig = CostEstimatorConfig(), **kwargs):
        pass

    @abstractmethod
    def compute_cost(self, **kwargs):
        pass

    pass
