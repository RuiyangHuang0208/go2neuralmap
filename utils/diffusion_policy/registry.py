from __future__ import annotations

from omegaconf import OmegaConf

from .algorithms import (
    CFGDDPMAlgorithm,
    CFGDDPMAlgorithmConfig,
    DDPMAlgorithm,
    DDPMAlgorithmConfig,
    GuidedJointDDPMAlgorithm,
    GuidedJointDDPMAlgorithmConfig,
    RegressionAlgorithm,
    RegressionAlgorithmConfig,
)


def build_algorithm(cfg_omega):
    name = str(cfg_omega.name)
    config = cfg_omega.config
    if name == "baseline":
        cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(RegressionAlgorithmConfig), config))
        return RegressionAlgorithm(cfg)
    if name == "ddpm":
        cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DDPMAlgorithmConfig), config))
        return DDPMAlgorithm(cfg)
    if name == "cfg_ddpm":
        cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(CFGDDPMAlgorithmConfig), config))
        return CFGDDPMAlgorithm(cfg)
    if name == "guided_joint_ddpm":
        cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(GuidedJointDDPMAlgorithmConfig), config))
        return GuidedJointDDPMAlgorithm(cfg)
    raise ValueError(f"Unknown diffusion policy algorithm: {name}")
