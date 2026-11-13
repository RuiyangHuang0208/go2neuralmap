# Residual Model Training Plan

## Goal

Implement a residual policy and an RL training pipeline that improves the
diffusion navigation policy through environment interaction.

The full navigation stack is:

1. Build/update a neural map and cost map from rollout experience.
2. Use A* on the cost map to produce navigation supervision.
3. Distill A* behavior into a diffusion velocity-command policy with
   supervised learning.
4. Freeze the diffusion policy and train a residual policy with PPO to adjust
   the diffusion policy output during interaction.

The residual policy should mainly compensate for failures of the frozen
diffusion policy, especially local obstacle avoidance and recovery behavior.
Goal-directed navigation remains primarily the responsibility of the diffusion
policy.

## Scope

In scope:

- Add a residual policy module.
- Add a high-level residual RL environment/wrapper.
- Add a PPO training entry point and configs.
- Reuse existing `rsl_rl` PPO implementation as much as possible.
- Reuse the existing diffusion observation/normalization contract.

Out of scope:

- Do not change existing Go2 locomotion behavior.
- Do not retrain or update the diffusion policy during residual RL.
- Do not change existing diffusion training behavior.
- Do not make unrelated refactors in `legged_gym` or `rsl_rl`.

## Target Files / Areas

Likely new areas:

```text
utils/residual_policy/
  __init__.py
  command_mixer.py
  env.py
  observation.py
  residual_actor_critic.py

scripts/experiments/train_residual_policy.py
scripts/experiments/configs/residual_training.yaml
```

Likely integration areas:

```text
legged_gym/legged_gym/envs/go2
rsl_rl/rsl_rl
scripts/experiments/modules/vel_command_generator/diffusion_policy.py
utils/diffusion_policy
```

Prefer adding new modules/wrappers over modifying existing behavior.

## Core Design Decisions

### Residual Action Horizon

Use `action_horizon`, not the full diffusion `prediction_horizon`.

Current default:

```text
diffusion prediction_horizon = 10
execution/action_horizon    = 2
command_dt                  = 0.5 s
main_loop_hz                = 50 Hz
```

The residual policy only needs to output corrections for commands that will
actually be executed before replanning. Therefore:

```text
residual_horizon = execution_horizon = 2
```

### Residual Action Space

The residual must correct heading as well as planar velocity.

Per residual command:

```text
[delta_vx, delta_vy, delta_heading]
```

Full PPO action:

```text
action_dim = residual_horizon * 3
```

For `residual_horizon = 2`:

```text
action_dim = 6
```

The environment reshapes the PPO action as:

```text
[num_envs, residual_horizon, 3]
```

### Command Composition

The frozen diffusion policy produces a base command sequence:

```text
base_commands: [num_envs, residual_horizon, 3]
```

where each command is:

```text
[vx, vy, heading]
```

The residual policy produces normalized residual actions. The final command is:

```python
final_vx = base_vx + residual_vxy_scale * action_delta_vx
final_vy = base_vy + residual_vxy_scale * action_delta_vy
final_heading = wrap_pi(base_heading + residual_heading_scale * action_delta_heading)
```

Then clamp:

```python
final_vx, final_vy = clamp_by_velocity_limits(...)
final_heading = wrap_pi(final_heading)
```

Initial recommended config:

```yaml
residual_vxy_scale: 0.3
residual_heading_scale: 0.4
max_forward_speed: 1.5
max_lateral_speed: 0.5
```

### Frozen Diffusion Stem and Encoder

During residual RL:

- Freeze the full diffusion model.
- Freeze the diffusion stem / shared condition encoder.
- Train only the residual model and PPO critic.

The diffusion model is used for:

- producing the base velocity-command sequence
- producing the frozen encoded condition feature used by the residual policy

No gradients should flow into diffusion modules.

## Observation

The residual policy observation should reuse the diffusion policy conditioning
contract and normalization.

The diffusion inference batch already contains:

```text
map_crop
uncertainty_crop
goal_delta
current_latent
current_heading
```

Use the frozen diffusion stem/encoder to compute:

```text
encoded_feature = frozen_diffusion_encoder(batch)
```

Then build residual actor observation:

```text
residual_obs = concat(
    encoded_feature,
    current_latent,
    goal_direction_local
)
```

Recommended `goal_direction_local`:

```text
[unit_goal_dir_x_local, unit_goal_dir_y_local, clipped_goal_distance]
```

The goal direction is computed from:

```text
goal_xy - current_xy
```

and rotated into the robot local frame using the current base heading.

Normalization:

- Use the same `goal_scale`, `velocity_scale`, uncertainty transform, crop size,
  and latent conventions as the diffusion policy.
- Do not create a separate residual normalization unless training shows a clear
  need.

## Reward

The residual policy is meant to improve local behavior while relying on the
diffusion policy for goal-directed navigation. Therefore reward should support
two variants.

### Variant A: Sparse Residual Reward

This variant matches the intended objective most directly.

```text
r = reached_goal_reward
  - collision_penalty
  - residual_penalty
  - termination_penalty
```

Terms:

- `reached_goal_reward`: positive reward when distance to goal is within
  `goal_tolerance`.
- `collision_penalty`: penalty when configured key bodies collide with walls or
  obstacles.
- `residual_penalty`: small penalty on residual magnitude.
- `termination_penalty`: penalty for fall, invalid state, or early failure.

Recommended initial values:

```yaml
goal_tolerance: 0.35
reached_goal_reward: 10.0
collision_penalty: 5.0
termination_penalty: 10.0
residual_penalty_scale: 0.01
```

Collision detection should reuse `GO2NM` key-body contact logic:

```python
env.get_directional_collision_flags()
env._cost_collision()
```

### Variant B: Shaped Debug Reward

This variant is for easier diagnosis and more stable early training.

```text
r = progress_reward
  + reached_goal_reward
  - collision_penalty
  - residual_penalty
  - termination_penalty
```

where:

```python
progress_reward = previous_distance_to_goal - current_distance_to_goal
```

This reward should be configurable and easy to disable. It is not the primary
research target, but it is useful for debugging the residual training pipeline.

## Episode Termination

Terminate the high-level residual episode when:

- the goal is reached
- the robot falls or `GO2NM` terminates
- max residual episode length is reached
- optionally, repeated collision count exceeds a threshold

When the goal is reached:

```text
done = True
```

The next reset samples a new start/goal pair through the existing environment
and terrain mechanisms.

## High-Level RL Step Semantics

One PPO environment step corresponds to one complete residual command sequence
execution.

For default settings:

```text
residual_horizon = 2
command_dt = 0.5 s
main_loop_hz = 50 Hz
```

One PPO step executes:

```text
2 commands * 0.5 s/command * 50 low-level steps/s = 50 low-level control steps
```

Execution loop:

1. Build diffusion conditioning batch from current neural map and goal.
2. Run frozen diffusion policy to get `base_commands`.
3. Build residual observation using frozen diffusion encoder.
4. PPO actor outputs normalized residual sequence.
5. Compose `final_commands = base_commands + scaled_residual`.
6. Execute each command for `command_dt`.
7. During execution, run frozen locomotion policy at the normal control rate.
8. Update neural map and cost map after each low-level step.
9. Accumulate collision information and reward terms.
10. Return next residual observation, reward, done, info.

## Residual Model Structure

First implementation: MLP actor-critic.

Actor hidden dims:

```text
[512, 512, 256]
```

Critic hidden dims:

```text
[512, 512, 256]
```

Actor output:

```text
residual_horizon * 3
```

For PPO compatibility, either:

1. Use existing `rsl_rl.modules.ActorCritic` with configured hidden dims and
   `env.num_actions = residual_horizon * 3`.
2. Add `ResidualActorCritic` only if we need custom action squashing or custom
   feature handling that cannot be cleanly represented as flat observations.

Preferred first step: use the existing `ActorCritic`.

## PPO Integration

The residual training environment/wrapper should expose the standard `VecEnv`
fields expected by `OnPolicyRunner`:

```python
num_envs
num_obs
num_privileged_obs
num_actions
max_episode_length
episode_length_buf
reset()
step(actions)
get_observations()
get_privileged_observations()
```

This keeps changes to `rsl_rl` minimal.

If the wrapper can match the current `VecEnv` contract, do not modify PPO.

## Implementation Phases

### Phase 1: Observation and Command Composition

- Extract reusable code for building diffusion inference batch.
- Add a residual observation builder.
- Add command mixer:
  - input: base command sequence and residual action
  - output: clipped final command sequence
- Add unit tests for shapes and heading wrapping.

### Phase 2: Residual RL Environment Wrapper

- Implement high-level wrapper around `GO2NM`.
- Load frozen locomotion policy.
- Load frozen diffusion policy.
- Freeze diffusion model and encoder.
- Execute `residual_horizon * command_dt` per PPO step.
- Return sparse reward variant first.

### Phase 3: PPO Training Entry Point

- Add `scripts/experiments/train_residual_policy.py`.
- Add Hydra/YAML config.
- Register config fields for:
  - diffusion checkpoint
  - locomotion checkpoint
  - reward variant
  - residual scales
  - horizon and timing
  - PPO params

### Phase 4: Reward Variant B and Logging

- Add shaped debug reward.
- Log:
  - goal success rate
  - collision rate
  - mean residual norm
  - mean distance-to-goal
  - mean progress
  - base command vs final command statistics

### Phase 5: Smoke Tests

- Shape-only test for residual observation.
- Shape-only test for command mixer.
- One-env rollout smoke test in headless mode.
- PPO short run smoke test if Isaac Gym is available.

## Open Implementation Notes

- The first implementation should train only one residual model over all envs.
- Multi-env support is required because PPO needs parallel rollouts.
- The wrapper should support `num_envs > 1`; avoid logic that only reads env 0.
- Existing `DiffusionPolicyVelCommandGenerator` currently has several env-0
  assumptions. The residual wrapper should either generalize the needed pieces
  or implement its own vectorized batch builder.
- The residual policy should not directly access cost-map ground truth as an
  observation unless explicitly added later as an ablation. Cost/collision
  should be used for reward and logging.

## Current Confirmed Decisions

- Residual corrects heading.
- One PPO step executes the full residual `action_horizon`.
- Diffusion stem and encoder are fixed.
- Only the residual model and PPO critic are trained.
- Reaching the goal terminates the residual episode.
- Implement both sparse and shaped reward variants.
- Keep residual penalty.
