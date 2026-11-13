# Position Navigation State Transitions and Recovery Mode Report

## Scope

This report describes the state transition and recovery-mode logic implemented in:

```text
scripts/experiments/modules/vel_command_generator/position_navigation.py
```

The module implements a waypoint-based velocity command generator for position navigation. It plans waypoints on the current cost map, follows them with a position controller, periodically replans, and enters a recovery mode when replanning fails.

Related task terminal bookkeeping is handled by:

```text
scripts/experiments/modules/vel_command_generator/navigation_task_evaluator.py
```

## High-Level Runtime Structure

`PositionNavigation.compute_vel_cmd()` creates one command per environment:

```text
commands: [num_envs, 4]
```

For each environment, `_compute_vel_cmd_for_env()` performs the full navigation update:

1. Read current robot local XY position.
2. Convert current XY to cost-map grid coordinates.
3. Detect environment reset or non-local grid jump.
4. Update trajectory cost.
5. Advance waypoint index if the active waypoint is reached.
6. Update task terminal status through `NavigationTaskEvaluator`.
7. Refresh task and navigation state if the task terminates.
8. Dispatch command generation according to `navigation_mode_by_env`.
9. Store per-env diagnostic info.

The dispatch point is:

```python
if self.navigation_mode_by_env[env_id] == NavigationMode.PLANNING_FAILURE_RECOVERY:
    env_cmd = self._compute_recovery_mode_cmd(env_id, current_xy, target_xy)
else:
    env_cmd = self._compute_normal_mode_cmd(env_id, current_xy, target_xy)
```

## Per-Environment State

The module keeps independent navigation state for every Isaac Gym environment:

- `goal_xy_by_env`: current goal for each environment
- `waypoints_by_env`: planned waypoint list for each environment
- `waypoint_idx_by_env`: active waypoint index
- `traj_cost_by_env`: accumulated traversal cost along the executed path
- `last_grid_xy_by_env`: previous grid coordinate used for cost accumulation and reset detection
- `navigation_mode_by_env`: current navigation mode
- `recovery_step_count_by_env`: number of steps spent in recovery mode
- `step_count_by_env`: normal-mode step counter used for periodic replanning
- `reached_goal_count_by_env`: success counter used for optional terrain reset
- `info_by_env`: diagnostic info exported by `get_info()`

The state is initialized by `_init_per_env_state()` and refreshed by `init_state()`.

## Navigation Modes

The explicit mode enum has two values:

```python
class NavigationMode(Enum):
    NORMAL = "normal"
    PLANNING_FAILURE_RECOVERY = "planning_failure_recovery"
```

### `NORMAL`

This is the default mode. In normal mode, the system follows the current waypoint path and periodically replans from the current pose.

The command path is:

```text
NORMAL
  -> _compute_normal_mode_cmd()
  -> optional periodic replan
  -> _compute_controller_cmd()
  -> position_controller.compute_command()
```

### `PLANNING_FAILURE_RECOVERY`

This mode is entered when a periodic replan fails. In recovery mode, the system keeps producing a command while retrying planning every `planning_failure_replan_interval` recovery steps.

The command path is:

```text
PLANNING_FAILURE_RECOVERY
  -> _compute_recovery_mode_cmd()
  -> optional retry replan
  -> fallback behavior selected by planning_failure_callback
```

## Initialization and Reanchoring Transitions

Several events reinitialize or reanchor navigation state and force the mode back to `NORMAL`.

### Initial State

`init_state()` runs once over all environments:

```text
init_state
  -> _refresh_navigation_from_current_pose(..., resample_goal=True)
  -> navigation_mode = NORMAL
  -> recovery_step_count = 0
```

`_refresh_navigation_from_current_pose()`:

- stores the current grid as `last_grid_xy_by_env`
- clears accumulated trajectory cost
- resets `step_count_by_env`
- samples or restores the goal through `NavigationTaskEvaluator`
- starts a new task timer
- plans waypoints from current pose to goal
- resets `waypoint_idx_by_env` to `0`

### Environment Reset Detection

During `_compute_vel_cmd_for_env()`, if `env.reset_buf[env_id]` is true:

```text
reset_buf detected
  -> _refresh_navigation_from_current_pose(..., resample_goal=True)
  -> navigation_mode = NORMAL
  -> recovery_step_count = 0
```

This handles external simulator resets by discarding stale waypoints and planning from the reset pose.

### Non-Local Grid Jump Detection

The module also detects discontinuities in the robot grid position:

```python
not self._is_cardinal_adjacent_or_same(last_grid_xy, current_grid_xy)
```

The allowed movement between consecutive calls is:

- same cell
- one cell in X
- one cell in Y

Diagonal or larger jumps are treated as non-local jumps. On such a jump:

```text
non-local grid jump
  -> _refresh_navigation_from_current_pose(..., resample_goal=True)
  -> navigation_mode = NORMAL
  -> recovery_step_count = 0
```

This is effectively another reset/reanchoring mechanism. It prevents the command generator from continuing with stale path state after teleportation or large state discontinuities.

## Waypoint Progression

Waypoint advancement happens before mode-specific command generation.

The current path is considered finished when:

```text
waypoints is None
or waypoint_idx >= len(waypoints)
```

Otherwise, the active waypoint is:

```text
target_xy = waypoints[waypoint_idx]
```

If the robot is close enough to the active waypoint:

```text
norm(current_xy - target_xy) <= cfg.waypoint_tolerance
```

the waypoint index is advanced. If the new index is still valid, the next waypoint becomes the target. If the index reaches the end, `reached_goal` becomes true.

Important detail: task success is driven by waypoint completion through:

```python
reached_goal_override=bool(reached_goal)
```

passed into `NavigationTaskEvaluator.update_env()`. Therefore, this module treats completion of the planned waypoint list as task success, instead of relying only on Euclidean distance to `goal_xy`.

## Task Terminal Transitions

After waypoint progression, `_compute_vel_cmd_for_env()` updates task status:

```python
task_update = self.task_evaluator.update_env(
    env_id=env_id,
    current_xy=current_xy,
    dt=float(getattr(self.env, "dt", 0.0)),
    reached_goal_override=bool(reached_goal),
)
```

The task can become terminal because of:

- success: waypoint list completed
- timeout: configured time or step limit reached before success

On terminal:

1. Log trajectory cost.
2. Reset `traj_cost_by_env`.
3. Increment `reached_goal_count_by_env` if success.
4. Optionally reset the environment or terrain.
5. Refresh navigation from the current pose.
6. Reset mode to `NORMAL` in the no-env-reset branch, and during terrain reset.

The reset decision is:

```text
should_reset_env =
    (reached_goal and reset_env_flag)
    or (timeout and timeout_action == "env_reset")
```

When `should_reset_env` is false, the task is reset in-place:

```text
task terminal without env reset
  -> _refresh_navigation_from_current_pose(...)
  -> navigation_mode = NORMAL
  -> recovery_step_count = 0
```

When `should_reset_env` is true, the environment is reset by `_reset_env()` or `_terrain_reset_env()`, then navigation is refreshed from the reset pose.

## Normal Mode Replanning

Normal mode uses `step_count_by_env` to trigger periodic replanning:

```python
if step_count > 0 and step_count % replan_interval == 0:
    _attempt_replan_from_current_pose(...)
```

If replanning succeeds:

```text
new waypoints accepted
waypoint_idx = 0
navigation_mode = NORMAL
recovery_step_count = 0
step_count = 0
continue with controller command
```

If replanning fails:

```text
_handle_planning_failure(...)
step_count = 0
fallback behavior may return immediate command
```

After the replan check, normal mode increments `step_count_by_env` and computes the controller command against the active target.

## Planning Failure Detection

Planning is attempted in `_plan_waypoints()`:

```text
planner_map = context.cost_map.get_planner_map(env_id)
start_grid = clamp(xy_to_grid(start_xy))
goal_grid = clamp(xy_to_grid(goal_xy))
result = context.planner.plan(start_grid, goal_grid, planner_map)
```

Planning is considered failed when:

```python
result.state != PlannerState.SUCCESS or result.path is None
```

Possible planner failure states include:

- `NO_PATH`
- `INVALID_START`
- `INVALID_GOAL`
- `INVALID_MAP`

On failure, `_plan_waypoints()` returns `None`. Periodic normal-mode replanning then calls `_handle_planning_failure()`.

## Entering Recovery Mode

`_handle_planning_failure()` always switches the environment into recovery mode:

```python
self.navigation_mode_by_env[env_id] = NavigationMode.PLANNING_FAILURE_RECOVERY
self.recovery_step_count_by_env[env_id] = 0
```

Then it selects one of two configured callbacks:

```text
cfg.planning_failure_callback
```

Supported values:

- `keep_current`
- `backward_keep_heading_bypass_refiner`

Unknown values raise `ValueError`.

## Recovery Strategy: `keep_current`

With:

```text
planning_failure_callback = "keep_current"
```

entering recovery does not immediately override the command. `_handle_planning_failure()` returns `None`, so normal mode falls through and computes a command using the existing waypoint list.

During recovery mode:

```text
recovery_step_count += 1
retry replan every planning_failure_replan_interval steps
if retry succeeds:
    return to NORMAL
else:
    keep following current waypoints with the position controller
```

The fallback command is:

```text
_compute_controller_cmd(env_id, current_xy, target_xy)
```

This strategy assumes the previous path is still useful while the planner waits for the cost map to become plannable again.

## Recovery Strategy: `backward_keep_heading_bypass_refiner`

With:

```text
planning_failure_callback = "backward_keep_heading_bypass_refiner"
```

the recovery command bypasses the waypoint position controller and directly issues a body-frame backward velocity:

```text
env_cmd[0] = -planning_failure_backward_velocity
env_cmd[3] = current heading command
```

The heading command comes from `_get_current_heading_cmd()`:

- if `env.commands` has a heading channel, reuse `env.commands[env_id, 3]`
- otherwise compute current robot heading from `root_states[env_id, 3:7]`

During recovery, this same backward command is emitted between retry replans. The intent is to back the robot away from a potentially invalid or blocked planning state while preserving heading.

## Exiting Recovery Mode

Recovery exits only when `_attempt_replan_from_current_pose()` succeeds:

```text
new_waypoints != None
  -> waypoints_by_env = new_waypoints
  -> waypoint_idx = 0
  -> navigation_mode = NORMAL
  -> recovery_step_count = 0
```

In recovery mode, replanning is retried every:

```text
cfg.planning_failure_replan_interval
```

steps.

On successful retry, `_compute_recovery_mode_cmd()` sets:

```python
self.step_count_by_env[env_id] = 1
```

and immediately computes a normal controller command using the new path.

## State Transition Summary

```text
init_state
  -> NORMAL

external reset detected
  -> refresh navigation
  -> NORMAL

non-local grid jump detected
  -> refresh navigation
  -> NORMAL

NORMAL
  -> periodic replan succeeds
  -> NORMAL with new waypoints

NORMAL
  -> periodic replan fails
  -> PLANNING_FAILURE_RECOVERY

PLANNING_FAILURE_RECOVERY
  -> retry replan fails
  -> stay in PLANNING_FAILURE_RECOVERY

PLANNING_FAILURE_RECOVERY
  -> retry replan succeeds
  -> NORMAL with new waypoints

task success or timeout terminal
  -> reset or refresh task/navigation
  -> NORMAL
```

## Configuration Parameters Related to State and Recovery

- `replan_interval`: normal-mode replanning period
- `waypoint_tolerance`: distance threshold for advancing the active waypoint
- `reset_env_flag`: whether success should reset the environment
- `timeout_limit_s`: optional task timeout in seconds
- `timeout_limit_steps`: optional task timeout in steps
- `timeout_action`: either `task_reset` or `env_reset`
- `advance_goal_on_success`: whether to sample or advance the goal after success
- `advance_goal_on_timeout`: whether to sample or advance the goal after timeout
- `advance_goal_on_done`: forwarded to task evaluator for external reset handling
- `terrain_reset_after_goal_count`: success count threshold for terrain reset
- `terrain_reset_resample_goal`: whether terrain reset should resample goal
- `reset_maps_on_terrain_reset`: whether neural and cost maps are reset after terrain reset
- `map_least_count`: resolution used for XY/grid conversion in this module
- `planning_failure_callback`: recovery strategy selector
- `planning_failure_backward_velocity`: backward speed used by backward recovery
- `planning_failure_replan_interval`: replanning retry period in recovery mode

## Diagnostics Exported in `info`

Each per-env info dictionary includes recovery and state fields:

- `planning_failure_callback`
- `navigation_mode`
- `recovery_step_count`
- `current_xy`
- `current_grid_xy`
- `goal_xy`
- `traj_cost`
- `reached_goal_count`
- task evaluator fields such as `task_terminal_flag`, `task_success_flag`, `task_timeout_flag`, and `task_terminal_reason`

`compute_vel_cmd()` stores the first environment info at `self.info` and also exposes all environments under:

```text
self.info["per_env"]
```

## Implementation Notes and Risks

- Recovery mode is entered only from periodic replanning failure. Initial planning failure inside `_refresh_navigation_from_current_pose()` produces `waypoints_by_env = None`, but does not directly switch to recovery mode.
- With `keep_current`, if `waypoints_by_env` is already `None`, the module marks `reached_goal = True` before command generation. This can cause the task evaluator to treat the task as successful even though planning failed.
- The stop and task-success semantics are waypoint-list based in this module. The task evaluator's geometric goal distance is bypassed by `reached_goal_override`.
- Non-local grid jump detection rejects diagonal grid transitions as discontinuities, because only same-cell and cardinal one-cell moves are accepted. If the robot can legitimately move diagonally by one grid cell between command updates, this logic will reanchor navigation.
- `_attempt_replan_from_current_pose()` returns to `NORMAL` only when a new path is found. Until then, recovery behavior is fully determined by `planning_failure_callback`.
