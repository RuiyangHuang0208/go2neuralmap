from .cost_estimator import CostEstimatorConfig, CostEstimator
from dataclasses import dataclass, field
from legged_gym.envs.go2.go2_nm_env import GO2NM
from loguru import logger
from typing import Optional

class EstimateFromEnvConfig(CostEstimatorConfig):
    pass

class EstimateFromEnv(CostEstimator):
    def __init__(
            self,
            cfg: EstimateFromEnvConfig = EstimateFromEnvConfig(),
            env_handle: Optional[GO2NM] = None,
            **kwargs):
        self.logger = logger.bind(module_name="CostEstim.Env")
        if env_handle is None:
            self.logger.error("env handler not passed in")
            raise ValueError("No env handle specified")
        self.env_handle: GO2NM = env_handle
        pass

    def compute_cost(self, **kwargs):
        # Returns cost retrieved from env, with shape=(num_envs,)
        return self.env_handle.step_cost
