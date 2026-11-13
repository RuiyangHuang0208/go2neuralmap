# RL Navigation Observation Construction

## Scope

This document describes how the high-level RL navigation observation is built for `utils/rl_nav_policy`.

The implementation lives in:

```text
utils/rl_nav_policy/observation.py
```

The same `RLNavObservationBuilder` is used by the training wrapper:

```text
utils/rl_nav_policy/env.py
```

and by runtime policy inference modules that load RL navigation policies.

## Inputs

Each observation is built from:

- `neural_map.maps`
- `neural_map.uncertainties` or `neural_map.get_aged_uncertainties()`
- robot local position: `env.root_local_pos[:, :2]`
- robot orientation: `env.root_states[:, 3:7]`
- robot velocity: `env.base_lin_vel`, `env.base_ang_vel`, or fallback slices from `env.root_states`
- current low-level command: `env.commands[:, :3]`
- high-level goal: `goal_xy`
- previous high-level command: `previous_command`

All final observation tensors are returned as `float32` on the configured RL device.

## Map Layouts

The observation builder supports two NeuralMap tensor layouts.

Directional map:

```text
neural_map.maps:          [B, X, Y, D, L]
neural_map.uncertainties: [B, X, Y, D]
```

Cell-wise map:

```text
neural_map.maps:          [B, X, Y, L]
neural_map.uncertainties: [B, X, Y]
```

Where:

- `B`: number of environments
- `X, Y`: map grid dimensions
- `D`: number of directional channels
- `L`: latent dimension per channel

For the default directional Bayesian NeuralMap:

```text
D = 4
L = long_latent_dim + short_latent_dim = 64
```

so the latent part contributes:

```text
4 * 64 = 256
```

features per map token.

## Token Dimension

Every map token has the same semantic structure:

```text
[latent, scaled_uncertainty, geometry]
```

The token dimension is:

```text
token_dim = num_channels * latent_dim + num_channels + 6
```

The final `6` geometry values are:

```text
rel_robot_x
rel_robot_y
dist_robot
rel_goal_x
rel_goal_y
dist_goal
```

For the default directional map:

```text
token_dim = 4 * 64 + 4 + 6 = 266
```

For a cell-wise map with latent dim `64`:

```text
token_dim = 1 * 64 + 1 + 6 = 71
```

## Local Crop Tokens

Local crop tokens are always built, including V1/V2 and V3.

The robot position is converted to a map grid index:

```text
x_idx = floor((x - x_min) / least_count)
y_idx = floor((y - y_min) / least_count)
```

The local crop is centered on that grid cell:

```text
crop_size x crop_size
```

With the default config:

```yaml
crop_size: 8
```

the local branch produces:

```text
local_tokens: [B, 64, token_dim]
```

Out-of-bounds cells are handled as follows:

- map latent values are padded with `0.0`
- uncertainty is padded with `uncertainty_max`

The uncertainty crop is transformed through:

```text
scale_uncertainty_crop(..., uncertainty_log_scale)
```

so the token receives scaled uncertainty rather than raw uncertainty.

## Local Token Geometry

Each local crop cell is converted back to a metric XY cell center:

```text
cell_x = x_min + (grid_x + 0.5) * least_count
cell_y = y_min + (grid_y + 0.5) * least_count
```

Then geometry is computed relative to the robot and the goal:

```text
rel_robot = (cell_xy - robot_xy) / goal_scale
dist_robot = norm(rel_robot)

rel_goal = (goal_xy - cell_xy) / goal_scale
dist_goal = clamp(norm(goal_xy - cell_xy) / goal_scale, 0, clipped_goal_distance)
```

Default scales:

```yaml
goal_scale: 12.0
clipped_goal_distance: 1.0
```

## V3 Global Pooled Map Tokens

V3 enables the global branch with:

```yaml
include_global_map: true
global_pool_size: 12
```

The full NeuralMap is pooled to:

```text
global_pool_size x global_pool_size
```

With the default map configuration:

```text
x_range: [-2.0, 10.0]
y_range: [-2.0, 10.0]
least_count: 0.25
```

the full map is:

```text
48 x 48
```

and V3 pools it to:

```text
12 x 12 = 144 global tokens
```

Each pooled token covers approximately:

```text
4 x 4 original cells = 1.0m x 1.0m
```

The V3 global branch produces:

```text
global_tokens: [B, 144, token_dim]
```

If `include_global_map` is `false`, the builder returns an empty global token tensor:

```text
[B, 0, token_dim]
```

This keeps the V1/V2 flat observation layout compatible.

## Global Pooling Rule

Raw uncertainty is first scaled in the same way as the local crop:

```text
scaled_uncertainty = scale_uncertainty_crop(raw_uncertainty)
```

Then confidence is computed as:

```text
confidence = clamp(1.0 - scaled_uncertainty, 0.0, 1.0)
```

Latents are pooled using confidence-weighted adaptive average pooling:

```text
pooled_latent =
  adaptive_avg_pool2d(latent * confidence)
  / clamp(adaptive_avg_pool2d(confidence), min=1e-6)
```

Scaled uncertainty is pooled with adaptive average pooling:

```text
pooled_uncertainty = adaptive_avg_pool2d(scaled_uncertainty)
```

For directional maps, confidence is expanded so each direction and latent channel is weighted independently.

## Global Token Geometry

Global pooled token geometry is not averaged from original local geometries.

Instead, the builder computes each pooled block center in map coordinates:

```text
pooled_center_x = x_min + pooled_grid_x * least_count
pooled_center_y = y_min + pooled_grid_y * least_count
```

Then it recomputes the same geometry features used by local tokens:

```text
rel_robot_x
rel_robot_y
dist_robot
rel_goal_x
rel_goal_y
dist_goal
```

This keeps local and global token semantics aligned while preserving the different spatial scales.

## Condition Vector

The condition vector is concatenated after the map tokens.

Default condition layout:

```text
[goal_direction_local, state_features, previous_command]
```

### Goal Direction

`goal_direction_local` has 3 values:

```text
local_goal_dir_x
local_goal_dir_y
clipped_goal_distance
```

The world-space goal direction is rotated into the robot heading frame using the yaw inferred from `root_states[:, 3:7]`.

### State Features

`state_features` has 8 values:

```text
base_lin_vel_x / velocity_scale
base_lin_vel_y / velocity_scale
base_ang_vel_z / yaw_rate_scale
sin(heading)
cos(heading)
current_command_vx / velocity_scale
current_command_vy / velocity_scale
current_command_yaw_rate / yaw_rate_scale
```

Default scales:

```yaml
velocity_scale: 1.5
yaw_rate_scale: 1.0
```

### Previous Command

If enabled:

```yaml
include_previous_command: true
```

the previous high-level command contributes 3 more values:

```text
previous_command_vx / velocity_scale
previous_command_vy / velocity_scale
previous_command_yaw_rate / yaw_rate_scale
```

Therefore, the default condition dimension is:

```text
condition_dim = 3 + 8 + 3 = 14
```

If `include_previous_command` is disabled:

```text
condition_dim = 3 + 8 = 11
```

## Final Flat Observation Layout

The final observation is a flat tensor:

```text
obs = concat(
  local_tokens.flatten(),
  global_tokens.flatten(),
  condition
)
```

Shape:

```text
[B, observation_dim]
```

Where:

```text
observation_dim =
  (local_num_tokens + global_num_tokens) * token_dim
  + condition_dim
```

## V1/V2 Default Shape

For V1/V2, `include_global_map=false`.

Default directional map:

```text
local_num_tokens = 8 * 8 = 64
global_num_tokens = 0
token_dim = 266
condition_dim = 14
```

Final observation dimension:

```text
64 * 266 + 14 = 17038
```

Flat layout:

```text
[local_tokens.flatten(), condition]
```

## V3 Default Shape

For V3:

```yaml
include_global_map: true
global_pool_size: 12
```

Default directional map:

```text
local_num_tokens = 8 * 8 = 64
global_num_tokens = 12 * 12 = 144
token_dim = 266
condition_dim = 14
```

Final observation dimension:

```text
(64 + 144) * 266 + 14 = 55342
```

Flat layout:

```text
[local_tokens.flatten(), global_tokens.flatten(), condition]
```

## Policy Consumption

### RLNavV2

`ActorCriticRLNavV2` expects:

```text
[local_tokens.flatten(), condition]
```

It reshapes:

```text
local_tokens: [B, crop_size * crop_size, token_dim]
condition:    [B, condition_dim]
```

The local tokens are projected and used as cross-attention key/value.

### RLNavV3

`ActorCriticRLNavV3` expects:

```text
[local_tokens.flatten(), global_tokens.flatten(), condition]
```

It reshapes:

```text
local_tokens:  [B, local_crop_size * local_crop_size, token_dim]
global_tokens: [B, global_pool_size * global_pool_size, token_dim]
condition:     [B, condition_dim]
```

The local crop remains the only attention key/value source:

```text
K,V = local_features
```

The global pooled map is encoded into `global_embedding` and only contributes to query generation:

```text
query = MLP([condition_feature, local_embedding, global_embedding])
```

The V3 encoder output is:

```text
[attended_local, local_embedding, global_embedding, condition_feature]
```

with shape:

```text
[B, 4 * encoder_embed_dim]
```

## Important Implementation Notes

- Observation construction runs under `torch.no_grad()`.
- NeuralMap tensors are detached before token construction.
- The global pooled branch is active only when `include_global_map=true`.
- V3 changes the observation and policy encoder, not the reward function.
- The training script writes `token_dim` and `condition_dim` from the constructed environment so policy dimensions match the selected NeuralMap layout.
