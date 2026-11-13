# RL Navigation Cell-Wise Bayesian Neural Map

## Summary

This change adds a cell-wise Bayesian Neural Map variant for the RL navigation policy ablation.

The baseline RL navigation config uses `directional_bayesian_neural_map`, where each map cell stores four direction-specific latent vectors and four uncertainty values. The new variant keeps the same Bayesian fusion and uncertainty aging idea, but stores one latent vector and one uncertainty value per cell.

This supports the experiment:

```text
directional Bayesian Neural Map
vs
cell-wise Bayesian Neural Map
```

with all other RL navigation settings kept aligned.

## Files Added

- `utils/neural_map/cellwise_bayesian_neural_map.py`
  - Defines `CellwiseBayesianNeuralMapCfg`.
  - Defines `CellwiseBayesianNeuralMap`.
  - Stores:

    ```text
    maps:          [num_envs, x, y, latent_dim]
    uncertainties: [num_envs, x, y]
    ```

  - Supports Bayesian update, uncertainty aging, reset, save, and load.
  - Exposes `long_maps` and `short_maps` as views into `maps`.

- `utils/neural_map/test_cellwise_bayesian_neural_map.py`
  - Covers tensor shapes, Bayesian updates, save/load, reset, and read-only aged uncertainty queries.

- `scripts/experiments/configs/rl_nav_training_cellwise_bayesian.yaml`
  - Adds a Hydra training config derived from `rl_nav_training`.
  - Overrides the Neural Map name to `cellwise_bayesian_neural_map`.
  - Keeps the same latent dimensions and Bayesian uncertainty parameters as the directional baseline.
  - Sets `ppo.policy.token_dim: 71`.

## Files Updated

- `utils/neural_map/registry.py`
  - Registers:

    ```text
    cellwise_bayesian_neural_map
    ```

- `utils/neural_map/__init__.py`
  - Exports `CellwiseBayesianNeuralMap` and `CellwiseBayesianNeuralMapCfg`.

- `utils/rl_nav_policy/observation.py`
  - `RLNavObservationBuilder` now supports both map layouts:

    ```text
    directional: maps.ndim == 5, uncertainties.ndim == 4
    cell-wise:   maps.ndim == 4, uncertainties.ndim == 3
    ```

  - Cell-wise maps are treated as a single-channel map during token construction.
  - `token_dim(neural_map)` now derives the token dimension from the actual map layout.

- `utils/rl_nav_policy/test_observation.py`
  - Adds coverage for cell-wise map observation layout.

## Observation Layout Impact

Directional baseline token dimension:

```text
4 * 64 latent + 4 uncertainty + 6 geometry = 266
```

Cell-wise variant token dimension:

```text
64 latent + 1 uncertainty + 6 geometry = 71
```

The flat PPO observation remains:

```text
[map_tokens.flatten(), condition]
```

The condition layout remains unchanged.

## Training

Example command:

```bash
python scripts/experiments/train_rl_nav_policy.py --config_name rl_nav_training_cellwise_bayesian
```

`scripts/experiments/train_rl_nav_policy.py` already updates `ppo.policy.crop_size`, `ppo.policy.token_dim`, and `ppo.policy.condition_dim` from the constructed wrapper before creating the PPO runner, so the policy dimensions stay aligned with the selected map layout.

## Verification

Commands run:

```bash
pytest -q utils/neural_map/test_cellwise_bayesian_neural_map.py utils/neural_map/test_neural_map_load.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_reward.py utils/rl_nav_policy/test_model.py utils/rl_nav_policy/test_command.py
pytest -q utils/rl_nav_policy
pytest -q utils/neural_map/test_cellwise_bayesian_neural_map.py utils/neural_map/test_neural_map_load.py utils/rl_nav_policy
```

Results:

- Focused Neural Map and RL navigation tests: 27 passed.
- Full `utils/rl_nav_policy` tests: 15 passed.

## Notes

- The existing directional Bayesian map path is preserved.
- `CellwiseBayesianNeuralMap.update()` accepts unused keyword arguments such as `root_velocity` so it remains compatible with `RLNavPolicyEnv`.
- `CellwiseBayesianNeuralMap.get_uncertainty(..., direction=...)` raises a `ValueError`, because this map stores one uncertainty value per cell rather than per direction.
- Existing unrelated untracked workspace items were not modified.
