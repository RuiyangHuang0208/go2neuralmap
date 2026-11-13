# Bezier Position Controller Velocity Generation

## Scope

This document describes the velocity generation method implemented in:

```text
utils/position_controller/position_controller_bezier.py
```

The controller receives the robot planar position, robot base orientation, target position, and optional waypoint context. It returns a `PositionControllerCommand`, where `command_xy` is the planar velocity command in the robot body frame.

## Core Idea

The Bezier position controller separates velocity generation into two parts:

- path direction: computed from the tangent of a cubic Bezier curve in the world frame
- speed magnitude: computed from target distance and scaled by heading error

Compared with directly commanding velocity along `target_xy - current_xy`, this controller uses waypoint context to build a smoother local path direction. The robot first aligns its heading with the Bezier tangent direction, then translates along that tangent.

## Inputs

The main entry point is:

```python
compute_command(current_xy, root_quat_xyzw, target_xy, **kwargs)
```

Required inputs:

- `current_xy`: current robot position in the world XY plane
- `root_quat_xyzw`: robot base orientation quaternion in XYZW format
- `target_xy`: final target or current task target in the world XY plane

Optional inputs passed through `kwargs`:

- `waypoints`: list of waypoint tensors
- `waypoint_idx`: active waypoint index

When waypoint information is available, the active waypoint is used as the end point of the current segment. Neighboring waypoints shape the Bezier tangent direction. When waypoint information is not available, the controller falls back to a single segment from `current_xy` to `target_xy`.

## Cubic Bezier Curve Construction

The controller first resolves four 2D points:

```text
prev_xy, start_xy, end_xy, next_xy
```

Their meanings are:

- `start_xy`: current robot position
- `end_xy`: active waypoint, or `target_xy` when no waypoint is active
- `prev_xy`: previous waypoint, or `start_xy` when unavailable
- `next_xy`: next waypoint, or `end_xy` when unavailable

These points are resolved by `_resolve_segment_points()`.

Then `_build_control_points()` constructs the cubic Bezier control points:

```text
P0 = start_xy
P1 = start_xy + tangent_scale * (end_xy - prev_xy)
P2 = end_xy   - tangent_scale * (next_xy - start_xy)
P3 = end_xy
```

The corresponding cubic Bezier curve is:

```text
B(u) = (1-u)^3 P0
     + 3(1-u)^2 u P1
     + 3(1-u) u^2 P2
     + u^3 P3
```

where:

```text
u in [0, 1]
```

The implementation does not directly track a position on this curve. Instead, it differentiates the curve and uses the derivative as the local path tangent direction.

## Bezier Derivative and Tangent Direction

The derivative of the cubic Bezier curve is:

```text
B'(u) = 3(1-u)^2 (P1 - P0)
      + 6(1-u)u (P2 - P1)
      + 3u^2 (P3 - P2)
```

This formula is implemented in `_eval_bezier_derivative()`.

The derivative sample location is configured by:

```text
cfg.derivative_eval_u
```

The value is clamped to `[0, 1]`. The default is `0.0`, so the velocity direction comes from the tangent at the start of the current segment:

```text
B'(0) = 3(P1 - P0)
```

After evaluating the derivative, the controller normalizes it:

```text
tangent_dir = velocity_world / norm(velocity_world)
```

If the tangent length is too small, the Bezier derivative is treated as degenerate. The implementation then falls back to the direct vector from the current position to the current segment end:

```text
velocity_world = end_xy - current_xy
```

This degeneracy check is controlled by:

```text
cfg.min_tangent_norm
```

## Target Heading

The Bezier tangent direction defines the desired robot heading:

```text
heading_target = atan2(tangent_dir_y, tangent_dir_x)
```

The current heading is computed by applying the robot quaternion to the forward vector `[1, 0, 0]`, then computing:

```text
current_heading = atan2(forward_y, forward_x)
```

The heading error is wrapped to `[-pi, pi]`:

```text
heading_error = wrap_to_pi(heading_target - current_heading)
```

## Stop Condition Near the Target

Before building the Bezier command, the controller computes the distance from `current_xy` to `target_xy`:

```text
distance = norm(target_xy - current_xy)
```

If:

```text
distance <= cfg.goal_tolerance
```

the controller returns zero velocity and sets the target heading to the current heading.

Note that this stop distance is computed against `target_xy`, which is not necessarily the active waypoint selected by `waypoint_idx`.

## Turn-in-Place Condition

If the heading error is too large:

```text
abs(heading_error) >= cfg.turn_in_place_angle
```

the controller returns zero translational velocity. This makes the robot rotate toward the Bezier tangent before translating.

The returned command still keeps the Bezier-generated `heading_target`, so the downstream controller can continue correcting yaw.

## Speed Magnitude

The nominal speed magnitude is generated by proportional distance control:

```text
commanded_speed = cfg.position_gain * distance
```

It is then clamped to:

```text
0 <= commanded_speed <= cfg.max_speed
```

Here, `distance` is also measured from `current_xy` to `target_xy`.

## Heading-Based Speed Scaling

Heading alignment is computed from the heading error:

```text
heading_alignment = clamp(cos(heading_error), 0, 1)
```

When the heading error is large but still below the turn-in-place threshold:

```text
abs(heading_error) >= cfg.lateral_zero_angle
```

the speed scale is capped by `cfg.heading_slowdown`:

```text
speed_scale = clamp(heading_alignment, 0, cfg.heading_slowdown)
```

When the heading error is smaller:

```text
abs(heading_error) < cfg.lateral_zero_angle
```

the speed scale is clamped between `cfg.heading_slowdown` and `1.0`:

```text
speed_scale = clamp(heading_alignment, cfg.heading_slowdown, 1.0)
```

The final world-frame velocity is:

```text
velocity_world = tangent_dir * commanded_speed * speed_scale
```

Therefore, the Bezier derivative only determines velocity direction. The actual speed magnitude is determined by target distance, speed limits, and heading alignment.

## World-to-Body Velocity Conversion

The generated velocity is first represented as a world-frame 3D vector with zero vertical component:

```text
velocity_world_3d = [velocity_world_x, velocity_world_y, 0]
```

It is then converted into the robot body frame with inverse quaternion rotation:

```python
quat_rotate_inverse(root_quat_xyzw, velocity_world_3d)
```

Only the planar part is kept:

```text
command_xy = body_velocity[:2]
```

Component-wise limits are then applied:

```text
command_xy[0] in [-cfg.max_speed, cfg.max_speed]
command_xy[1] in [-cfg.max_lateral_speed, cfg.max_lateral_speed]
```

Finally, the planar speed norm is limited again:

```text
norm(command_xy) <= cfg.max_speed
```

## Configuration Parameters

Bezier-specific parameters:

- `derivative_eval_u`: Bezier derivative sampling location, clamped to `[0, 1]`
- `tangent_scale`: scale factor used to place the inner control points `P1` and `P2`
- `min_tangent_norm`: degeneracy threshold for the Bezier tangent vector

Shared position-controller parameters:

- `goal_tolerance`: stopping threshold near `target_xy`
- `position_gain`: proportional gain from distance to speed
- `max_speed`: maximum planar speed
- `max_lateral_speed`: maximum body-frame lateral speed
- `heading_slowdown`: speed scaling parameter caused by heading error
- `turn_in_place_angle`: heading error threshold for turn-in-place behavior
- `lateral_zero_angle`: heading error threshold between normal slowdown and stronger slowdown

## Implementation Flow

The velocity generation flow in `compute_command()` is:

1. Normalize input tensor shapes.
2. Compute target distance and current robot heading.
3. Return zero velocity if the robot is within `goal_tolerance`.
4. Resolve the current Bezier segment context from `waypoints` and `waypoint_idx`.
5. Construct cubic Bezier control points.
6. Compute and normalize the Bezier derivative to obtain the world-frame tangent direction.
7. Compute `heading_target` from the tangent direction.
8. Return zero translational velocity if heading error exceeds `turn_in_place_angle`.
9. Compute the speed scale from heading error.
10. Generate world-frame velocity along the Bezier tangent.
11. Rotate world-frame velocity into the robot body frame.
12. Apply longitudinal, lateral, and planar speed limits.
13. Return `PositionControllerCommand`.

## Notes

- The controller is local and reactive: since `start_xy` is the current robot position, a new local Bezier segment is rebuilt on every call.
- The default `derivative_eval_u = 0.0` uses the tangent at the start of the current segment. Increasing this value makes the direction depend more on later parts of the curve.
- Waypoint context only affects the Bezier tangent shape. The stop condition and nominal speed magnitude are still based on `target_xy`.
- The output `command_xy` is a body-frame planar velocity, while `heading_target` is a world-frame yaw angle.
