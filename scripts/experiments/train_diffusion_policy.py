#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime

import hydra
from loguru import logger
from omegaconf import OmegaConf

from utils.diffusion_policy import train_from_config


@hydra.main(config_path="configs", config_name="diffusion_training", version_base="1.1")
def main(cfg_omega) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg_omega.run_timestamp = timestamp
    OmegaConf.resolve(cfg_omega)
    logger.info("Composed config:\n{}", OmegaConf.to_yaml(cfg_omega, resolve=False))
    train_from_config(cfg_omega)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
