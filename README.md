# Terrain-Aware Quadruped Navigation with Proprioceptive Neural Maps

By [Liwei Lu], 2026 (Master's Thesis)

[Website](TODO) [Paper](TODO) [Video](TODO)

# News
  * 2026-02-03. Repository initialized and structure defined.

# Content Table

  * Installation
  * Usage
  * Citation and Contact

# Installation

This repository contains three main components:

- `legged_gym`: training framework
- `rsl_rl`: PPO algorithm implementation
- `neural_map`: core feature of this repository

1. Create and activate a conda environment:

    ```
    conda create -n go2_neural_map python=3.8
    conda activate go2_neural_map
    ```

2. Download dependencies:
    - Download `isaacgym` from [official link](https://developer.nvidia.com/isaac-gym)
    
    
3. Install local packages:
    ```
    cd isaacgym/python && pip install -e .
    cd rsl_rl && pip install -e . && cd ..
    cd legged_gym && pip install -e . && cd ..
    cd neural_map && pip install -e . && cd ..
    ```

# Usage

- Training entry points live under `legged_gym`.
- PPO implementation lives under `rsl_rl`.
- Proprioceptive neural map models and utilities live under `neural_map`.

Please add run scripts to `scripts/` for common commands as they stabilize.

# Citation and Contact

If you find this work useful, please cite:

    @mastersthesis{author2026neuralmap,
      title={Terrain-Aware Quadruped Navigation with Proprioceptive Neural Maps},
      author={Author, Name},
      year={2026},
      school={Your University}
    }

Contact: TODO
