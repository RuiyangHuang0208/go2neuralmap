# Reset / Goal Refactor Report

## Summary

This change introduces explicit reset result/event plumbing for GO2 terrain navigation while preserving the existing legged_gym `reset_idx()` convention used by `step()`.

The main addition is a typed `ResetResult` payload and new env-level APIs for:

- `resample_goals`: sample a new goal without moving the robot.
- `robot_reset`: reset the robot to the stored task start without changing terrain/start/goal.
- `task_reset`: keep terrain fixed, sample a new start, optionally sample a new goal, and reset robot state.
- `terrain_reset`: sample or assign terrain, sample start/goal, and reset robot state.

`reset_idx()` still runs inside `step()` as before and remains the automatic bottom-level reset path.

## Files Changed

- `utils/reset_events.py`
  - Added `ResetResult`.

- `legged_gym/legged_gym/envs/go2/go2_rough_env.py`
  - Added task start/goal state buffers.
  - Added `last_reset_result`.
  - Added explicit reset APIs.
  - Factored shared reset helpers for start sampling, env origin update, robot/buffer reset, root-local refresh, callbacks, and reset result construction.
  - Preserved `reset_idx()` behavior while making it produce `last_reset_result`.

- `scripts/experiments/modules/vel_command_generator/diffusion_policy.py`
  - Reached-goal behavior now uses goal-only resampling by default.
  - When `reset_env_flag=True`, uses `task_reset()` when available.

- `scripts/experiments/modules/vel_command_generator/residual_diffusion_policy.py`
  - Same goal-only resampling path for reached-goal cases.
  - Uses `task_reset()` when reset is explicitly requested.

- `scripts/experiments/modules/vel_command_generator/position_navigation.py`
  - Goal sampling now uses `resample_goals()` when available.
  - Explicit env reset uses `task_reset()` when available.

- `utils/residual_policy/env.py`
  - Task reset and goal sampling use the env reset APIs when available.
  - Robot-only reset delegates to `robot_reset()` when available.
  - Synchronizes wrapper task start into env task-start state.

- `utils/terrain_loader/test_terrain_loader.py`
  - Added goal sequential sampling wraparound coverage.

- `utils/test_reset_events.py`
  - Added basic `ResetResult` payload test.

## Semantics

`reset_buf` / returned `dones` remains the legged_gym bottom-level signal: a step transition terminated and `reset_idx()` has already run inside `step()`.

The new APIs are for upper-level task/controller decisions:

- Goal reached with continuous navigation: call `resample_goals()` and invalidate planner/diffusion state.
- Need a new start on same terrain: call `task_reset()`.
- Need a new terrain episode: call `terrain_reset()` and use callbacks or upper-layer policy to reset neural/cost maps.
- Need to recover robot pose only: call `robot_reset()`.

## Verification

Commands run:

```bash
python3 -m py_compile legged_gym/legged_gym/envs/go2/go2_rough_env.py
python3 -m py_compile scripts/experiments/modules/vel_command_generator/diffusion_policy.py
python3 -m py_compile scripts/experiments/modules/vel_command_generator/residual_diffusion_policy.py
python3 -m py_compile scripts/experiments/modules/vel_command_generator/position_navigation.py utils/residual_policy/env.py utils/reset_events.py
```

Using the project conda environment:

```bash
/home/liwei/F/003LinuxSoftwares/miniconda3/envs/go2_neural_map/bin/python -m pytest -q utils/terrain_loader/test_terrain_loader.py
/home/liwei/F/003LinuxSoftwares/miniconda3/envs/go2_neural_map/bin/python -m pytest -q utils/test_reset_events.py
```

Results:

- `utils/terrain_loader/test_terrain_loader.py`: 8 passed.
- `utils/test_reset_events.py`: 1 passed.

Isaac Gym import verification required writable extension cache:

```bash
LD_LIBRARY_PATH=/home/liwei/F/003LinuxSoftwares/miniconda3/envs/go2_neural_map/lib \
TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
PYTHONPATH=legged_gym:isaacgym/python \
/home/liwei/F/003LinuxSoftwares/miniconda3/envs/go2_neural_map/bin/python - <<'PY'
from legged_gym.envs.go2.go2_rough_env import ResetResult, GO2Rough
print(ResetResult.__name__, GO2Rough.__name__)
PY
```

Result:

```text
ResetResult GO2Rough
```

## Follow-Up Risks

- `ResetResult` is now available, but not every upper-level module consumes `last_reset_result` yet.
- Map reset policy is still task-owned and should be centralized in a later pass.
- Some cost map subclasses reset only running state rather than full map values; this should be reviewed before terrain-level map reset policy is finalized.

