from .estimate_from_env import EstimateFromEnvConfig, EstimateFromEnv
from .estimate_from_critic import EstimateFromCritic, EstimateFromCriticConfig
from .estimate_from_cost_pred_nn import EstimateFromCostPredNN, EstimateFromCostPredNNCfg

CostEstimatorRegistry = {
    "estimate_from_env"    : (EstimateFromEnv, EstimateFromEnvConfig()),
    "estimate_from_critic" : (EstimateFromCritic, EstimateFromCriticConfig()),
    "estimate_from_cost_pred_nn" : (EstimateFromCostPredNN, EstimateFromCostPredNNCfg()),
}
