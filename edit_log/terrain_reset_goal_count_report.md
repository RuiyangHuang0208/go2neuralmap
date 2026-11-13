# Terrain Reset Goal Count Report

## Summary

This change adds an optional terrain-reset policy to `PositionNavigation`.

By default the feature is disabled. When enabled, each environment tracks how many goals it has reached. Once the per-env count reaches the configured threshold, the navigation module performs a terrain reset, resets the related maps, clears the count, and replans from the new pose.

The requested behavior is implemented as "trigger on the 10th reached goal" for the provided experiment config.

## Files Changed

- `scripts/experiments/modules/vel_command_generator/position_navigation.py`
  - Added terrain reset configuration fields:
    - `terrain_reset_after_goal_count`
    - `terrain_reset_resample_goal`
    - `reset_maps_on_terrain_reset`
  - Added per-env `reached_goal_count_by_env`.
  - Added `terrain_reset_flag` and `reached_goal_count` to `RunningInfo`.
  - Added helper methods for terrain reset, map reset, and threshold checking.
  - Goal-reaching logic now triggers `terrain_reset()` when the configured count threshold is reached.

- `utils/cost_map/ema_cost_map.py`
  - `reset()` now delegates to base `CostMap.reset()` so map values are reset to `cfg.init_value`.

- `utils/cost_map/transition_cost_map.py`
  - `reset()` now clears map values through base `CostMap.reset()` and then clears transition running state.

- `utils/cost_map/directional_ema_velocity_cost_map.py`
  - `reset()` now clears map values through base `CostMap.reset()` and then clears directional velocity state.

- `utils/cost_map/directional_ema_transition_cost_map.py`
  - `reset()` now clears map values through base `CostMap.reset()` and then clears directional transition state.

- `utils/cost_map/test_ema_cost_map.py`
  - Updated reset coverage to assert selected env map contents are cleared.

- `scripts/experiments/configs/exp/demo_plane_real_path_dataset_terrain_reset.yaml`
  - Added an experiment config based on `demo_plane_real_path_dataset`.
  - Enables terrain reset on the 10th reached goal.

## Configuration

Default `PositionNavigationConfig` behavior:

```python
terrain_reset_after_goal_count = 0
terrain_reset_resample_goal = True
reset_maps_on_terrain_reset = True
```

`terrain_reset_after_goal_count = 0` disables terrain reset. The new experiment config sets:

```yaml
modules:
  vel_command_generator:
    config:
      terrain_reset_after_goal_count: 10
      terrain_reset_resample_goal: true
      reset_maps_on_terrain_reset: true
```

## Runtime Semantics

For each env:

1. A reached goal increments `reached_goal_count_by_env[env_id]`.
2. If `terrain_reset_after_goal_count <= 0`, the old goal-reset behavior is preserved.
3. If the count reaches the configured threshold, `PositionNavigation`:
   - sets `reached_goal_flag = True`;
   - sets `terrain_reset_flag = True`;
   - calls `env.terrain_reset(...)`;
   - resets `neural_map` and `cost_map` for that env when `reset_maps_on_terrain_reset=True`;
   - clears the reached-goal count for that env;
   - returns navigation mode to `NORMAL`;
   - replans from the reset pose.

Cost map reset semantics are now consistent with neural map reset expectations: selected env map values are cleared inside the cost map implementation itself.

## Verification

Commands run:

```bash
pytest -q utils/cost_map
python -m py_compile scripts/experiments/modules/vel_command_generator/position_navigation.py
python -m py_compile utils/cost_map/ema_cost_map.py utils/cost_map/transition_cost_map.py utils/cost_map/directional_ema_velocity_cost_map.py utils/cost_map/directional_ema_transition_cost_map.py
```

Results:

- `utils/cost_map`: 21 passed.
- Python compile checks passed.

## Notes

- The terrain reset feature is intentionally opt-in to preserve existing experiment behavior.
- Existing untracked workspace items `data_sampling.sh` and `go2_smoke/` were not modified.
