from dataclasses import dataclass, field

from loguru import logger

from utils.cost_predict_nn.cost_predict_nn_sigle_latent import (
    Config, CostPredNNSingleLatent)

from .cost_estimator import CostEstimator, CostEstimatorConfig


@dataclass
class EstimateFromCostPredNNCfg(CostEstimatorConfig):
    nn_cfg: Config = field(default_factory=Config)


class EstimateFromCostPredNN(CostEstimator):
    def __init__(self, cfg: EstimateFromCostPredNNCfg, **kwargs):
        self.logger = logger.bind(module_name="CostEstimator.CostPredNN")

        self.cfg = cfg
        self.cost_pred_nn = CostPredNNSingleLatent(cfg.nn_cfg)
        self.cost_pred_nn.load_weights()
        self.cost_pred_nn.to(self.cfg.nn_cfg.device)
        pass

    def compute_cost(self, **kwargs):
        latent = kwargs.get("latent")
        if latent is None:
            self.logger.error("Priv Obs not passed")
            raise ValueError("Priv Obs not passed")
        cost = self.cost_pred_nn.inference(latent).squeeze(-1)
        return cost
