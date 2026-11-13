# Navigation Task Evaluation and Point-to-Point Baseline

## Summary

This change adds a shared navigation task evaluation helper for rollout-time velocity command generators and introduces a simple point-to-point navigation baseline.

The evaluation helper is currently integrated into:

- `rl_nav_policy`
- `position_navigation`

Keyboard input is intentionally excluded because it has no task-goal semantics. Diffusion-based generators are also left unchanged for now.

## Files Added

- `scripts/experiments/modules/vel_command_generator/navigation_task_evaluator.py`
  - Adds `NavigationTaskEvaluatorConfig`.
  - Adds `NavigationTaskEvaluator`.
  - Tracks per-env task state:
    - current goal
    - goal source
    - goal list index
    - reached goal count
    - task elapsed steps
    - task elapsed time
    - task start position
  - Supports:
    - `goal_source: sampled | user | list`
    - `goal_xy_list`
    - `timeout_limit_s`
    - `timeout_limit_steps`
    - `timeout_action: task_reset | env_reset`
    - `advance_goal_on_success`
    - `advance_goal_on_timeout`
    - `advance_goal_on_done`
  - Emits recorder-compatible task info:
    - `task_terminal_flag`
    - `task_success_flag`
    - `task_timeout_flag`
    - `task_terminal_reason`
    - `task_elapsed_time_s`
    - `task_elapsed_steps`
    - `task_start_xy`
    - `current_xy`
    - `goal_xy`
    - `distance_to_goal`

- `scripts/experiments/modules/vel_command_generator/point_to_point_navigation.py`
  - Adds `PointToPointNavigationConfig`.
  - Adds `PointToPointNavigation`.
  - Uses `position_controller.compute_command(...)` directly.
  - Does not use planner, cost map, or neural map for navigation decisions.
  - Reuses `NavigationTaskEvaluator` for goal lists, timeout, and task logging.

- `scripts/experiments/configs/exp/rl_nav_4.yaml`
  - Evaluation-focused variant of `rl_nav_3`.
  - Adds explicit timeout/task evaluation settings for `rl_nav_policy`.

- `scripts/experiments/configs/exp/demo_plane_9.yaml`
  - Evaluation-focused variant of `demo_plane_8`.
  - Uses `position_navigation` with shared task evaluation settings.

- `scripts/experiments/configs/exp/demo_plane_10.yaml`
  - Point-to-point baseline variant.
  - Inherits `demo_plane_9` settings but switches to:

    ```yaml
    modules:
      vel_command_generator:
        name: point_to_point_navigation
    ```

- `scripts/experiments/test_navigation_task_evaluator.py`
  - Adds unit coverage for list-goal cycling and timeout behavior.

## Files Updated

- `scripts/experiments/modules/vel_command_generator/rl_nav_policy.py`
  - Moves goal list handling, success detection, timeout detection, and task info emission to `NavigationTaskEvaluator`.
  - Keeps RL policy inference, action scaling, command hold behavior, and attention visualization unchanged.
  - `goal_tolerance: null` now falls back to the loaded RL nav training config's `rl_nav.reward.goal_tolerance`.

- `scripts/experiments/modules/vel_command_generator/position_navigation.py`
  - Adds shared task evaluation config fields.
  - Uses `NavigationTaskEvaluator` for goal source/list handling, timeout, and recorder info.
  - Keeps planner/waypoint-based success semantics.
  - Adds `waypoint_tolerance` to replace the previous hard-coded `0.25` waypoint threshold.
  - `reset_env_flag: false` now allows task reset without returning the robot to the start pose.

- `scripts/experiments/modules/vel_command_generator/registry.py`
  - Registers:

    ```text
    point_to_point_navigation
    ```

- `scripts/experiments/helpers/state_recorder/navigation_metrics_recorder.py`
  - Prioritizes shared task event fields when present.
  - Keeps fallback compatibility with old `reached_goal_flag` and env `done`.
  - Adds per-episode:
    - `task_duration_s`
  - Adds summary metrics:
    - `timeout_count`
    - `done_count`
    - `timeout_rate`
    - `done_rate`
    - `task_duration_mean_s`

- `scripts/experiments/helpers/state_recorder/test_navigation_metrics_recorder.py`
  - Adds coverage for timeout task logging and task duration summary.

## Evaluation Config Examples

`rl_nav_4.yaml`:

```yaml
modules:
  vel_command_generator:
    config:
      goal_tolerance: null
      timeout_limit_s: 30.0
      timeout_limit_steps: null
      timeout_action: task_reset
      advance_goal_on_success: true
      advance_goal_on_timeout: true
      advance_goal_on_done: true
```

`demo_plane_9.yaml`:

```yaml
modules:
  vel_command_generator:
    config:
      goal_source: list
      goal_xy_list:
        - [1.0, 4.0]
        - [7.0, 4.0]
      goal_tolerance: 0.35
      waypoint_tolerance: 0.25
      timeout_limit_s: 30.0
      timeout_action: task_reset
      advance_goal_on_success: true
      advance_goal_on_timeout: true
      advance_goal_on_done: true
```

For `position_navigation`, set:

```yaml
reset_env_flag: false
```

to switch to the next goal and replan from the current robot pose instead of resetting the robot to the start pose.

## Point-to-Point Baseline

`point_to_point_navigation` implements:

```text
current_xy + goal_xy -> position_controller -> env.commands
```

It is intended as a no-map baseline for simple point-to-point navigation. It does not use:

- `planner`
- `cost_map`
- `neural_map`
- waypoints
- replan interval

The task success condition is distance based:

```text
norm(current_xy - goal_xy) <= goal_tolerance
```

## Validation

Ran:

```text
python -m py_compile \
  scripts/experiments/modules/vel_command_generator/navigation_task_evaluator.py \
  scripts/experiments/modules/vel_command_generator/rl_nav_policy.py \
  scripts/experiments/modules/vel_command_generator/position_navigation.py \
  scripts/experiments/modules/vel_command_generator/point_to_point_navigation.py \
  scripts/experiments/modules/vel_command_generator/registry.py \
  scripts/experiments/helpers/state_recorder/navigation_metrics_recorder.py
```

Ran:

```text
pytest -q \
  scripts/experiments/test_navigation_task_evaluator.py \
  scripts/experiments/helpers/state_recorder/test_navigation_metrics_recorder.py
```

Result:

```text
5 passed
```

Checked Hydra composition:

```text
+exp=rl_nav_4
-> rl_nav_policy 30.0 list [[1.0, 4.0], [7.0, 4.0]]

+exp=demo_plane_9
-> position_navigation list [[1.0, 4.0], [7.0, 4.0]] 30.0 0.25

+exp=demo_plane_10
-> point_to_point_navigation list [[1.0, 4.0], [7.0, 4.0]] 30.0 False
```

Checked registry import with Isaac Gym imported first and `TORCH_EXTENSIONS_DIR=/tmp/torch_extensions`:

```text
True PointToPointNavigation
```
