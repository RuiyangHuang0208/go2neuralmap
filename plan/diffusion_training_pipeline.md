# Diffusion Training Pipeline Plan

## Overview

Implement a configurable training pipeline for command-sequence prediction from
DirectionalBayesianNeuralMap snapshots.

The pipeline will train on augmented datasets only. The smoke dataset under
`go2_smoke/` is the initial validation target.

## Problem Formulation

The model receives:

- an 8x8 local crop of the neural map centered on `start_grid`
- the matching 8x8 local crop of neural-map uncertainty
- goal position relative to `start_xy`
- current latent at `start_grid`, using all 4 directional latent channels
- current heading inferred from the first A* path edge

The model predicts an `n`-step command sequence.

The command convention is:

- `vx, vy`: local-frame planar velocity
- `heading`: global heading target
- heading is represented during training as `sin(heading), cos(heading)`

So the training target is:

```text
[vx, vy, sin(heading), cos(heading)]
```

At inference time, heading is reconstructed with:

```python
heading = atan2(sin_heading, cos_heading)
```

This avoids the angle-wrap discontinuity around `pi` and `-pi`.

## Dataset Contract

Only augmented datasets are supported by this pipeline. Required fields:

- `neural_map`: `[N, X, Y, 4, latent_dim]`
- `neural_map_uncertainty`: `[N, X, Y, 4]`
- `start_grid`: `[N, 2]`
- `start_xy`: `[N, 2]`
- `goal_xy`: `[N, 2]`
- `astar_path`: `[N, max_path_length, 2]`
- `astar_path_mask`: `[N, max_path_length]`
- `velocity_commands`: `[N, horizon, 3]`
- `velocity_command_mask`: `[N, horizon]`

The dataloader must lazy-load shard files to avoid keeping the full dataset in
memory. A dataset index should map each global sample index to `(shard_path,
local_index)`, and the implementation may cache the most recently used shards.
It should also provide a configurable `max_samples` option for fast smoke tests.

## Local Crop Convention

Default crop size is 8x8.

Because the crop size is even, the crop is centered with this convention:

```text
x: start_x - 4 ... start_x + 3
y: start_y - 4 ... start_y + 3
```

The `start_grid` cell is therefore located at crop index `(4, 4)`.

Out-of-map latent values are zero-padded. Out-of-map uncertainty values are
padded with the configured maximum/unknown uncertainty.

## Heading Convention

Heading inference should match the existing augmentation logic in
`utils/position_controller/velocity_sequence.py`.

For a path step `i`, the geometric heading is inferred from:

```text
astar_path[i + 1] - astar_path[i]
```

The initial conditioning heading uses the first valid edge:

```text
astar_path[1] - astar_path[0]
```

## Normalization

Recommended defaults:

- `goal_delta = (goal_xy - start_xy) / goal_scale`
- `vx, vy` divided by `velocity_scale`
- uncertainty transformed with `log1p`, clamped, then divided by an uncertainty scale
- heading represented as `sin/cos`
- neural latent used as-is initially

All normalization constants must be configurable with Hydra.

## Architecture

Implement a reusable command-policy package under `utils/`.

Recommended structure:

```text
utils/diffusion_policy/
  __init__.py
  dataset.py
  registry.py
  trainer.py
  models/
    __init__.py
    encoders.py
    baseline.py
    ddpm.py
  algorithms/
    __init__.py
    regression.py
    ddpm.py
```

### Shared Encoder

- CNN encoder for local neural-map crop and uncertainty
- MLP encoder for vector inputs:
  - normalized goal delta
  - current heading as sin/cos
  - 4-directional current latent flattened

### Baseline

Implement a deterministic sequence regressor first. It predicts the full
`[vx, vy, sin_heading, cos_heading]` sequence directly.

This baseline is needed to validate that the dataset and conditioning are
learnable before relying on diffusion metrics.

### DDPM

Implement an MLP DDPM denoiser over flattened command sequences. The denoiser is
conditioned on the shared encoder output and timestep embedding.

The DDPM algorithm should own:

- noise schedule
- training loss
- reverse sampling
- inference conversion back to `[vx, vy, heading]`

## Training Entry Point

Add an experiment entry point following existing conventions:

```text
scripts/experiments/train_diffusion_policy.py
```

Add Hydra config files under:

```text
scripts/experiments/configs/
```

The training script should support:

- algorithm selection: `baseline` or `ddpm`
- dataset path
- train/validation split
- batch size and dataloader workers
- model dimensions
- optimizer settings
- checkpoint interval
- optional wandb logging

Wandb must be optional. If disabled or unavailable, training should still run
with console logging and local checkpoints.

## Metrics

Track masked validation metrics:

- command-space training loss
- `vx, vy` MSE
- heading angular error after reconstructing heading with `atan2`
- optional generated sequence visualizations later

## Implementation Phases

1. Implement the lazy augmented dataset and batch transforms.
2. Implement the shared encoder and baseline sequence regressor.
3. Implement a generalized trainer and optional wandb logging.
4. Verify baseline overfit/smoke training on `go2_smoke`.
5. Add DDPM algorithm using the same dataset and encoder contract.
6. Verify DDPM smoke training and sampling.
