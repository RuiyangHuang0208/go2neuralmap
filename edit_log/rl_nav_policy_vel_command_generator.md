# RL Navigation Policy Velocity Command Generator

## Summary

This change integrates trained RL navigation policies into the existing `vel_command_generator` runtime path.

The implementation keeps `scripts/experiments/evaluate_rl_nav_policy.py` as a standalone evaluation CLI. The runtime integration is implemented as a new generator that only performs high-level policy inference and returns velocity commands to `starter.Runner`.

This avoids nesting `RLNavPolicyEnv.step()` inside the normal experiment loop, where locomotion stepping, Neural Map updates, CostMap updates, state recording, and dataset collection are already handled.

## Files Added

- `scripts/experiments/modules/vel_command_generator/rl_nav_policy.py`
  - Defines `RLNavPolicyVelCommandGeneratorConfig`.
  - Defines `RLNavPolicyVelCommandGenerator`.
  - Loads a trained RL navigation run from `run_dir/config.yaml`.
  - Selects the checkpoint with the largest `model_*.pt` step when `checkpoint: null`.
  - Builds observations with `RLNavObservationBuilder`.
  - Loads the PPO actor through `OnPolicyRunner` using a minimal inference adapter.
  - Converts normalized policy actions to `[vx, vy, yaw_rate]` with `scale_rl_nav_actions()`.
  - Returns env commands with the fourth command channel fixed to `0`.
  - Supports sampled and user goals through `set_goal_xy()` and `clear_user_goal()`.
  - Exposes goal markers and recorder-compatible runtime info.

## Files Updated

- `scripts/experiments/modules/vel_command_generator/registry.py`
  - Registers:

    ```text
    rl_nav_policy
    ```

- `scripts/experiments/builder.py`
  - `merge_env_config()` now optionally propagates:

    ```yaml
    env:
      commands:
        heading_command: false
        resampling_time: 1000000000.0
    ```

  - This allows experiment configs to put the low-level env into yaw-rate command mode before env construction.

## Runtime Control Frequency

The generator now controls high-level policy frequency through its own config, following the diffusion generator style:

```yaml
modules:
  vel_command_generator:
    name: rl_nav_policy
    config:
      run_dir: logs/rl_nav_policy/...
      command_dt: 0.25
      main_loop_hz: 50.0
```

`steps_per_command` is computed as:

```text
round(command_dt * main_loop_hz)
```

The policy is evaluated only when `command_step_index == 0`. Intermediate main-loop iterations reuse the previous command.

The generator still reads `rl_nav` from the training config for observation layout, action scaling, and related policy semantics, but `command_dt` and `main_loop_hz` for runtime action-hold behavior are intentionally controlled by the `vel_command_generator` config.

When the env resets or the goal is reached, the command cursor is reset so the next call samples a fresh policy action instead of holding a stale command.

## Yaw-Rate Command Mode

The RL navigation action format remains:

```text
[vx, vy, yaw_rate]
```

The generator forces:

```python
env.cfg.commands.heading_command = False
```

and, when enabled:

```python
env.cfg.commands.resampling_time = 1.0e9
```

This prevents the low-level env command resampler or heading controller from overwriting the RL policy's yaw-rate command.

The returned env command has shape matching `env.commands`. For four-channel env commands:

```text
commands[:, 0] = vx
commands[:, 1] = vy
commands[:, 2] = yaw_rate
commands[:, 3] = 0
```

## Goal Handling

The generator follows the same goal sampling path as the other velocity command generators:

- Prefer `env.resample_goals(env_ids)` when available.
- Fall back to `env.sample_goals(env_ids)[:, :2]`.
- Sync sampled goals back into `env.task_goal_xy` when the env exposes task-goal buffers.

The config supports:

```yaml
goal_source: sampled
resample_goal_flag: true
reset_env_flag: false
```

`goal_tolerance` defaults to the loaded training config's `rl_nav.reward.goal_tolerance` unless explicitly overridden in the generator config.

## Recorder Compatibility

`trajectory_recorder.py` expects velocity command generators to provide:

```text
current_xy
current_grid_xy
goal_xy
reached_goal_flag
traj_cost
```

The RL navigation generator now provides these fields through `get_info()`.

Additional diagnostic fields include:

```text
action
command
command_speed
yaw_rate_abs
sampled_action
command_step_index
steps_per_command
```

## Trajectory Cost Handling

`directional_ema_velocity_cost_map.get_traversal_cost()` only accepts cardinal adjacent grid transitions:

```text
(1, 0), (-1, 0), (0, 1), (0, -1)
```

During continuous locomotion, a robot can cross a grid corner between two recorder samples, producing diagonal deltas such as:

```text
(1, -1)
```

The generator now only accumulates `traj_cost` when the grid transition is cardinal adjacent. Diagonal or multi-cell jumps re-anchor `last_grid_xy` without calling `get_traversal_cost()`, preventing runtime failures from recorder-side auxiliary cost accounting.

## Example Config

```yaml
modules:
  vel_command_generator:
    name: rl_nav_policy
    config:
      run_dir: logs/rl_nav_policy/2026-05-24_04-34-46_stage3
      checkpoint: null
      device: cuda
      command_dt: 0.25
      main_loop_hz: 50.0
      resample_goal_flag: true
      reset_env_flag: false
      goal_source: sampled
```

## Verification

Commands run:

```bash
python -m py_compile scripts/experiments/modules/vel_command_generator/rl_nav_policy.py scripts/experiments/modules/vel_command_generator/registry.py scripts/experiments/builder.py
env TORCH_EXTENSIONS_DIR=/tmp/torch_extensions python -c "import isaacgym; import sys; sys.path.insert(0, 'scripts/experiments'); from modules.vel_command_generator.registry import VelCommandGeneratorRegistry; cls, cfg = VelCommandGeneratorRegistry['rl_nav_policy']; print(cls.__name__, cfg.__class__.__name__)"
```

Results:

- Python compile checks passed.
- Registry import resolved `RLNavPolicyVelCommandGenerator`.

## Notes

- Full IsaacGym simulation was not run in the sandbox because the environment reported no CUDA runtime / NVML.
- Existing unrelated untracked workspace files were not modified.
