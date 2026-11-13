# Position Controller Module

## Overview
- Module path: `utils.position_controller`
- Purpose: convert a target 2D position in terrain/local coordinates into locomotion commands for the robot root frame.
- Main use case: target-following test scripts that need heading-aware velocity commands without walking backward when the goal is behind the robot.

## Core Behavior
- Computes the target heading from the vector pointing from the current position to the target.
- Converts position error from terrain/local coordinates into the robot root frame using the current root quaternion in `xyzw` ordering.
- Uses a turn-first policy:
  - if the goal is almost behind the robot, command zero planar velocity and let the heading controller rotate first,
  - if the goal is moderately misaligned, reduce forward motion and suppress lateral motion,
  - if the goal is roughly in front, command forward-dominant motion.
- Prevents backward walking by clamping the forward body-frame command to be nonnegative.

## Public API
- `PositionControllerConfig`
  - stores controller hyperparameters such as goal tolerance, speed limits, and heading thresholds.
- `PositionControllerCommand`
  - stores computed command outputs and diagnostic terms such as distance and heading error.
- `compute_turn_first_position_command(current_xy, root_quat_xyzw, target_xy, cfg)`
  - returns a `PositionControllerCommand` for a single robot.
- `sample_target_xy(...)`
  - samples a random target in a configurable 2D range, optionally enforcing a minimum distance from the current position.

## Coordinate Conventions
- `current_xy` and `target_xy` are 2D positions in terrain/local coordinates.
- `root_quat_xyzw` is the robot root quaternion in `xyzw` ordering.
- Returned `command_xy` is in the robot root frame and is suitable for `env.commands[env_id, :2]`.
- Returned `heading_target` is an absolute heading target and is suitable for `env.commands[env_id, 3]`.

## Design Notes
- The module is environment-agnostic: it does not depend on a specific env class and only needs the current position, root quaternion, and target position.
- The implementation uses Isaac Gym quaternion utilities and `wrap_to_pi` from `legged_gym`.
- All comments and public-facing descriptions should remain in English.
