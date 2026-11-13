# RL Nav Repeated Collision Penalty

## Summary

This change adds an optional RL navigation reward term that penalizes repeated collisions at the same local-map location.

The feature is disabled by default. When enabled, each environment keeps a collision heatmap in the same local coordinate frame as the Neural Map. A collision first queries the heatmap around the current robot cell, applies an extra penalty based on prior collision density, then writes the new collision into the heatmap.

## Files Changed

- `utils/rl_nav_policy/env.py`
  - Added `RLNavRepeatedCollisionPenaltyConfig`.
  - Added `reward.repeated_collision_penalty` config under `RLNavRewardConfig`.
  - Added per-env `repeated_collision_map` state.
  - Added collision local-position to grid-index conversion using `neural_map.cfg.x_range`, `y_range`, and `least_count`.
  - Added `_repeated_collision_penalty()`:
    - returns zero when disabled;
    - queries prior collision count in a configurable cell radius;
    - clamps and normalizes count by `max_count`;
    - scales by `penalty_scale`;
    - supports optional heatmap decay;
    - updates the heatmap after querying, so the first collision at a site is not treated as repeated.
  - Added `step_repeated_collision_penalty` to step metrics.
  - Clears repeated-collision heatmaps on map reset, not on ordinary episode buffer reset.

- `utils/rl_nav_policy/test_reward.py`
  - Updated reward tests for the new `_compute_reward()` input.
  - Added coverage for same-cell repeated collision penalties.
  - Added coverage for neighbor-radius penalties.
  - Added coverage that ordinary episode resets keep the heatmap while map resets clear it.

- `scripts/experiments/configs/rl_nav_training.yaml`
  - Added disabled default config:

```yaml
repeated_collision_penalty:
  enabled: false
  penalty_scale: 1.0
  radius_cells: 1
  max_count: 3.0
  decay: 1.0
```

- `scripts/experiments/configs/rl_nav_training_repeated_collision.yaml`
  - New training config based on `rl_nav_training_cellwise_bayesian`.
  - Enables repeated collision penalty.
  - Writes runs under `logs/rl_nav_policy/${run_timestamp}_repeated_collision`.

## Runtime Semantics

For each high-level RL step:

1. The wrapper collects whether any low-level step collided.
2. If repeated-collision penalty is enabled, collided envs query their local collision heatmap at the current root-local XY position.
3. The queried window is controlled by `radius_cells`.
4. The extra penalty is:

```text
penalty_scale * clamp(prior_window_collision_count, 0, max_count) / max_count
```

5. The current collision cell is incremented after computing the penalty.
6. The heatmap is kept across task resets that preserve the map and is cleared on map reset.

## Configuration

Use the new config directly:

```bash
python scripts/experiments/train_rl_nav_policy.py \
  --config_name rl_nav_training_repeated_collision
```

The main tuning knobs are:

- `penalty_scale`: maximum extra penalty from repeated local collisions.
- `radius_cells`: spatial tolerance around the current collision cell.
- `max_count`: count at which the penalty saturates.
- `decay`: heatmap retention per high-level step; `1.0` keeps memory until map reset.

## Verification

Commands run:

```bash
python -m py_compile utils/rl_nav_policy/env.py utils/rl_nav_policy/test_reward.py
pytest -q utils/rl_nav_policy/test_reward.py
pytest -q utils/rl_nav_policy
python -c "import hydra; from omegaconf import OmegaConf; from utils.rl_nav_policy.env import RLNavEnvConfig; hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1'); cfg=hydra.compose(config_name='rl_nav_training_repeated_collision', overrides=['+run_timestamp=test']); OmegaConf.resolve(cfg); rl=OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(RLNavEnvConfig()), cfg.rl_nav)); print(rl.reward.repeated_collision_penalty.enabled, rl.reward.repeated_collision_penalty.penalty_scale, rl.reward.repeated_collision_penalty.radius_cells, cfg.training.output_dir)"
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy/test_reward.py`: 6 passed.
- `utils/rl_nav_policy`: 21 passed.
- Hydra config composition printed:

```text
True 1.0 1 logs/rl_nav_policy/test_repeated_collision
```

## Notes

- Existing training configs keep previous behavior because the new reward term defaults to disabled.
- The repeated-collision memory is intentionally separate from Neural Map latent state, so this reward can be tuned or disabled without changing observation contents.
