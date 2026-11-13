# RL Navigation Policy Initial Implementation

## Background

This change introduces a standalone RL navigation policy that directly maps Neural Map observations, current state, and goal information to velocity commands.

Unlike `utils/residual_policy`, the new policy does not depend on a frozen diffusion policy and does not output a residual correction. It is intended to learn navigation behavior through environment interaction.

The first implementation follows these decisions:

- Action is a single low-frequency command, not a command sequence.
- Action format is `[vx, vy, yaw_rate]`.
- Observation uses a robot-centered Neural Map crop.
- CostMap is not used as policy input.
- The new module lives under `utils/rl_nav_policy/`.

## Files Added

- `utils/rl_nav_policy/__init__.py`
- `utils/rl_nav_policy/command.py`
  - Defines `RLNavCommandConfig`.
  - Converts normalized RL actions to clipped velocity commands.
  - Supports asymmetric forward/backward speed limits.
- `utils/rl_nav_policy/observation.py`
  - Defines `RLNavObservationConfig` and `RLNavObservationBuilder`.
  - Builds robot-centered Neural Map crop tokens.
  - Adds uncertainty, relative-to-robot geometry, and relative-to-goal geometry to each map token.
  - Adds goal direction, current state features, and optional previous command as condition features.
- `utils/rl_nav_policy/model.py`
  - Defines `RLNavMapEncoder`.
  - Uses condition-derived query features with cross attention over map tokens.
  - Adds a pooled/global map-token context branch.
- `utils/rl_nav_policy/env.py`
  - Defines `RLNavPolicyEnv`, a `VecEnv` wrapper for PPO training.
  - Executes a frozen locomotion policy under low-frequency high-level commands.
  - Updates Neural Map online from locomotion latents.
  - Implements shaped goal-progress reward.
  - Does not use diffusion policy or CostMap.
- `rsl_rl/rsl_rl/modules/actor_critic_rl_nav.py`
  - Defines `ActorCriticRLNav`.
  - Splits flat PPO observations into map tokens and condition features.
  - Uses separate actor and critic map encoders.
- `scripts/experiments/train_rl_nav_policy.py`
  - Training entrypoint for the new policy.
  - Builds env, Neural Map, locomotion model, wrapper, and PPO runner.
  - Forces yaw-rate command mode before env construction.
- `scripts/experiments/configs/rl_nav_training.yaml`
  - Base Hydra config for RL navigation training.
- `utils/rl_nav_policy/test_command.py`
- `utils/rl_nav_policy/test_observation.py`
- `utils/rl_nav_policy/test_model.py`

## Files Updated

- `rsl_rl/rsl_rl/modules/__init__.py`
  - Exports `ActorCriticRLNav`.
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - Imports `ActorCriticRLNav` so the existing `eval(policy_class_name)` path can instantiate it.

## Observation Layout

The wrapper exposes a flat observation to PPO:

```text
[map_tokens.flatten(), condition]
```

Each map token contains:

```text
directional neural latent
scaled directional uncertainty
dx_robot
dy_robot
dist_robot
dx_goal
dy_goal
dist_goal
```

The condition feature contains:

```text
goal_direction_local: [local_x, local_y, clipped_distance]
state features: [base_vx, base_vy, base_yaw_rate, sin_heading, cos_heading, current_cmd]
previous command, optional
```

This handles goals outside the crop by keeping the crop local while still encoding goal direction and per-cell relative-to-goal geometry.

## Action and Command Semantics

The policy action is normalized and has shape `[num_envs, 3]`:

```text
[vx, vy, yaw_rate]
```

`RLNavCommandConfig` scales it into env commands:

```text
commands[:, 0] = vx
commands[:, 1] = vy
commands[:, 2] = yaw_rate
```

The training script and config set:

```yaml
env:
  commands:
    heading_command: false
    resampling_time: 1000000000.0
```

`heading_command: false` is required because the low-level env otherwise overwrites `commands[:, 2]` from a target heading controller.

## Reward

The initial shaped reward is:

```text
+ progress toward goal
+ reached-goal bonus
- collision penalty
- low-level termination penalty
- time penalty
- command magnitude penalty
- command smoothness penalty
```

Sparse reward is still supported through the `variant` field, but the default config uses shaped progress reward.

## Training

Example command:

```bash
python scripts/experiments/train_rl_nav_policy.py --config_name rl_nav_training --num_envs 16
```

The PPO config uses:

```yaml
ppo:
  runner:
    policy_class_name: ActorCriticRLNav
```

The training script updates `crop_size`, `token_dim`, and `condition_dim` from the constructed wrapper before creating the runner, so the policy dimensions stay aligned with the Neural Map configuration.

## Verification

Commands run:

```bash
python -m py_compile utils/rl_nav_policy/command.py utils/rl_nav_policy/observation.py utils/rl_nav_policy/model.py utils/rl_nav_policy/env.py rsl_rl/rsl_rl/modules/actor_critic_rl_nav.py scripts/experiments/train_rl_nav_policy.py
pytest -q utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_model.py
python -c "from rsl_rl.runners import OnPolicyRunner; from rsl_rl.modules import ActorCriticRLNav; print('ok')"
python -c "import hydra; from omegaconf import OmegaConf; hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1'); cfg=hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test']); OmegaConf.resolve(cfg); print(cfg.ppo.runner.policy_class_name, cfg.env.commands.heading_command)"
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 4 passed.

### Map Out-of-Bounds Handling

During RL navigation training, exploration can drive the robot outside the configured Neural Map range even when the nominal maze is enclosed. `GridMap._xy_to_indices()` intentionally raises on out-of-range coordinates, which previously crashed training during Neural Map update.

`RLNavPolicyEnv` now handles this at the wrapper level:

- Checks `env.root_local_pos[:, :2]` against `neural_map.cfg.x_range/y_range` after every low-level step.
- Marks out-of-range envs with `extras["episode"]["map_out_of_bounds"]`.
- Skips map writes for those envs by masking them during Neural Map update.
- Restores any snapshotted map state for masked envs.
- Treats out-of-bounds as a terminal failure when `terminate_on_map_out_of_bounds=True`.

The default RL navigation config now includes:

```yaml
rl_nav:
  terminate_on_map_out_of_bounds: true
```

This keeps strict map indexing semantics while preventing a single exploratory failure from killing the whole PPO run.

### Collection Profiling

`RLNavPolicyEnv` now records lightweight timing diagnostics in `extras["episode"]` so the PPO runner can log them through TensorBoard or WandB.

Config:

```yaml
rl_nav:
  profiling_enabled: true
```

Logged keys include:

```text
profile_step_total_s
profile_command_s
profile_locomotion_s
profile_locomotion_s_per_low_step
profile_env_step_s
profile_env_step_s_per_low_step
profile_map_update_s
profile_map_update_s_per_low_step
profile_obs_refresh_s
profile_collision_check_s
profile_post_step_bookkeeping_s
profile_low_level_reset_s
profile_done_reset_s
```

CUDA synchronization is used around timed regions when the wrapper device is CUDA, so the numbers reflect actual GPU work instead of asynchronous launch time. The profiling is intended for bottleneck diagnosis and can be disabled with `profiling_enabled: false` after tuning.

### Batched Observation Crop

Profiling showed `profile_obs_refresh_s` was a major collection-time bottleneck. `RLNavObservationBuilder` now builds map tokens with batched indexing instead of looping over environments in Python.

The new batched path:

- Builds `[num_envs, crop_size, crop_size]` x/y index grids once.
- Gathers Neural Map values from `[N, X, Y, 4, D]` into `[N, C, C, 4, D]`.
- Gathers uncertainty values from `[N, X, Y, 4]` into `[N, C, C, 4]`.
- Preserves old padding semantics:
  - map values outside the grid are padded with `0.0`;
  - uncertainty outside the grid is padded with `uncertainty_max`.
- Builds relative-to-robot and relative-to-goal geometry tokens in batch.
- Gathers current directional latent values in batch.

Test coverage was added to compare batched map tokens against the old single-env reference crop/token builder, including boundary padding behavior.

Additional verification:

```bash
python -m py_compile utils/rl_nav_policy/observation.py utils/rl_nav_policy/env.py rsl_rl/rsl_rl/runners/on_policy_runner.py
pytest -q utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 5 passed.

### Goal Distance Sampling Filter

`RLNavPolicyEnv` now supports wrapper-level distance filtering for goal sampling. This is intended as the first curriculum mechanism without changing the lower-level terrain loader schema.

Config:

```yaml
rl_nav:
  goal_sampling:
    min_distance: 0.0
    max_distance: null
    max_attempts: 20
    fallback: accept_last
```

Behavior:

- If `min_distance > 0` or `max_distance` is set, and the env exposes `terrain_loader` plus `env_terrain_map`, the wrapper samples candidate goals directly from `terrain_loader.sample_goals(...)`.
- Candidates are filtered by distance from the current `root_local_pos`.
- Accepted goals are written to `RLNavPolicyEnv.goal_xy`.
- Accepted goals are also synchronized to `env.task_goal_xy` and `env.task_goal_valid` when those fields exist.
- This avoids repeated calls to `env.resample_goals(...)` during rejection sampling, so failed attempts do not repeatedly mutate env task-goal state.
- `fallback: accept_last` accepts the last candidate after `max_attempts` if no candidate satisfies the range.
- `fallback: keep_previous` preserves the previous wrapper goal after failed filtering.

Example curriculum overrides:

```bash
python scripts/experiments/train_rl_nav_policy.py \
  --config_name rl_nav_training \
  rl_nav.goal_sampling.min_distance=1.0 \
  rl_nav.goal_sampling.max_distance=3.0
```

Additional verification:

```bash
python -m py_compile utils/rl_nav_policy/env.py utils/rl_nav_policy/test_goal_sampling.py
pytest -q utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
python -c "import hydra; from omegaconf import OmegaConf; hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1'); cfg=hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test','rl_nav.goal_sampling.min_distance=1.0','rl_nav.goal_sampling.max_distance=3.0']); OmegaConf.resolve(cfg); print(cfg.rl_nav.goal_sampling.min_distance, cfg.rl_nav.goal_sampling.max_distance)"
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 7 passed.

### Training Resume Support

`scripts/experiments/train_rl_nav_policy.py` now supports loading a PPO checkpoint before training:

```bash
--resume_checkpoint /path/to/model_3000.pt
--resume_optimizer
```

By default, `--resume_checkpoint` loads only the actor-critic weights and current iteration metadata. `--resume_optimizer` also restores the optimizer state. For curriculum transitions such as stage0 to stage1, the recommended default is to omit `--resume_optimizer` so Adam state from the easier task does not over-constrain adaptation.

Example stage transition:

```bash
python scripts/experiments/train_rl_nav_policy.py \
  --config_name rl_nav_training \
  --my_exp_name training/rl_nav/stage1 \
  --resume_checkpoint logs/rl_nav_policy/<stage0_run>/model_3000.pt
```

Additional verification:

```bash
python -m py_compile scripts/experiments/train_rl_nav_policy.py
python - <<'PY'
import hydra
from omegaconf import OmegaConf
hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1')
for stage in ['training/rl_nav/stage0', 'training/rl_nav/stage1']:
    cfg = hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test', f'+exp={stage}'])
    OmegaConf.resolve(cfg)
    print(stage, cfg.env.terrain.num_rows, cfg.env.terrain.num_cols, cfg.rl_nav.goal_sampling.max_distance, cfg.ppo.policy.crop_size)
PY
pytest -q utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
```

Results:

- Python compile checks passed.
- Stage0/stage1 compose checks passed with matching policy structure.
- `utils/rl_nav_policy` tests: 7 passed.

### Terrain Resampling Flags

`RLNavPolicyEnv` now supports optional terrain-id resampling at task reset points.

New config:

```yaml
rl_nav:
  terrain_resampling:
    on_task_success: false
    on_map_reset: false
    on_timeout: false
    on_collision_done: false
```

When a flag is enabled and the wrapped env exposes `terrain_reset(...)`, `_reset_task(..., resample_terrain=True)` calls:

```python
env.terrain_reset(env_ids, resample_goal=False)
```

instead of:

```python
env.task_reset(env_ids, resample_goal=False)
```

The wrapper still samples goals itself afterward, so the RL nav goal-distance filter remains active.

Stage1 currently enables terrain resampling only at map reset:

```yaml
rl_nav:
  terrain_resampling:
    on_task_success: false
    on_map_reset: true
    on_timeout: false
    on_collision_done: false
```

This means one map episode can accumulate several goals on the same terrain, and the terrain id changes when `map_reset_success_threshold` is reached.

Additional verification:

```bash
python -m py_compile utils/rl_nav_policy/env.py utils/rl_nav_policy/test_terrain_resampling.py scripts/experiments/train_rl_nav_policy.py
pytest -q utils/rl_nav_policy/test_terrain_resampling.py utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
python - <<'PY'
import hydra
from omegaconf import OmegaConf
hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1')
cfg = hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test', '+exp=training/rl_nav/stage1'])
OmegaConf.resolve(cfg)
print(cfg.rl_nav.terrain_resampling.on_task_success, cfg.rl_nav.terrain_resampling.on_map_reset, cfg.rl_nav.terrain_resampling.on_timeout, cfg.rl_nav.terrain_resampling.on_collision_done)
PY
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 9 passed.
- Stage1 terrain resampling check printed `False True False False`.

### Per-Terrain Success Logging

`RLNavPolicyEnv` now logs terminal success statistics grouped by `env.env_terrain_map`.

For each terrain id with at least one done env in the current rollout, the wrapper adds:

```text
Episode/terrain_<id>_success_rate
Episode/terrain_<id>_episode_count
```

Example for stage1 with two loaded terrain instances:

```text
Episode/terrain_0_success_rate
Episode/terrain_0_episode_count
Episode/terrain_1_success_rate
Episode/terrain_1_episode_count
```

These are terminal episode metrics, not per-step densities. `terrain_<id>_episode_count` is useful for checking whether a terrain received enough completed episodes in the logged rollout before trusting its success rate.

Follow-up update:

- If `env.terrain_loader.terrain_instances[terrain_id].name` is available, metric names now include the sanitized terrain name.
- Non-alphanumeric characters in terrain names are replaced with `_`.
- If the terrain name is missing, the metric falls back to the id-only form.

Example named terrain metrics:

```text
Episode/terrain_0_d1_plane_success_rate
Episode/terrain_0_d1_plane_episode_count
Episode/terrain_1_d1_block_forest_success_rate
Episode/terrain_1_d1_block_forest_episode_count
```

Additional verification:

```bash
python -m py_compile utils/rl_nav_policy/env.py utils/rl_nav_policy/test_terrain_metrics.py
pytest -q utils/rl_nav_policy/test_terrain_metrics.py utils/rl_nav_policy/test_terrain_resampling.py utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 11 passed.
- Hydra override check printed `1.0 3.0`.

### Evaluation Script

Added `scripts/experiments/evaluate_rl_nav_policy.py` for deterministic evaluation of trained RL navigation checkpoints.

Behavior:

- Loads `config.yaml` from `--run_dir`.
- Applies evaluation-only overrides:
  - `--num_envs`
  - `--device`
  - `--terrain_yaml`
- Forces yaw-rate command mode:
  - `env_cfg.commands.heading_command = False`
  - `env_cfg.commands.resampling_time = 1.0e9`
- Disables training loggers and profiling during evaluation.
- Rebuilds `RLNavPolicyEnv`, `ActorCriticRLNav`, and PPO runner from the saved config.
- Loads the selected checkpoint with `load_optimizer=False`.
- Uses `act_inference()` for deterministic actions.
- Can draw current goal markers in the IsaacGym viewer.
- Runs until `--episodes` completed episodes or `--max_steps` high-level steps.
- Prints JSON summary metrics and optionally saves them with `--output`.

Example:

```bash
python scripts/experiments/evaluate_rl_nav_policy.py \
  --run_dir logs/rl_nav_policy/2026-05-19_18-01-23 \
  --checkpoint logs/rl_nav_policy/2026-05-19_18-01-23/model_500.pt \
  --num_envs 16 \
  --episodes 100 \
  --headless
```

Viewer mode with goal markers:

```bash
python scripts/experiments/evaluate_rl_nav_policy.py \
  --run_dir logs/rl_nav_policy/2026-05-19_18-01-23 \
  --checkpoint logs/rl_nav_policy/2026-05-19_18-01-23/model_500.pt \
  --num_envs 4 \
  --episodes 20 \
  --render \
  --draw_goals
```

Goal marker options:

```bash
--no_draw_goals
--goal_marker_radius 0.2
```

If `--checkpoint` is omitted, the script loads the latest `model_*.pt` in `--run_dir`.

Additional verification:

```bash
python -m py_compile scripts/experiments/evaluate_rl_nav_policy.py
pytest -q utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
```

Results:

- Python compile check passed.
- `utils/rl_nav_policy` tests: 7 passed.

### Logging Semantics Fix

Training logs now distinguish step-level diagnostics from terminal episode statistics.

Before this change, `RLNavPolicyEnv` returned all diagnostics under `infos["episode"]` at every high-level step. `OnPolicyRunner` therefore logged values such as `Episode/success` as a per-step reached-goal density, not as an episode success rate. This made successful short-distance training look much worse than deterministic evaluation.

Updated behavior:

- `infos["episode"]` contains only done-env terminal metrics:
  - `success_rate`
  - `collision_rate`
  - `map_out_of_bounds_rate`
  - `final_distance_to_goal`
  - `timeout_rate`
  - `low_level_done_rate`
  - `collision_done_rate`
- `infos["step"]` contains per-step diagnostics:
  - `step_reached_goal_rate`
  - `step_collision_rate`
  - `step_map_out_of_bounds_rate`
  - `step_distance_to_goal`
  - `step_progress`
  - `step_positive_progress`
  - `step_command_speed`
  - `step_yaw_rate_abs`
  - `step_command_delta_norm`
  - `step_profile_*`

`OnPolicyRunner` now logs both `infos["episode"]` and `infos["step"]`, while skipping empty tensors when no env finished in the current rollout.
The runner also aggregates the union of keys across mixed episode/step info dictionaries and skips missing keys, so sparse terminal metrics and dense step metrics can coexist in the same rollout buffer.

Expected WandB/TensorBoard names:

```text
Episode/success_rate
Episode/final_distance_to_goal
Episode/step_reached_goal_rate
Episode/step_distance_to_goal
Episode/step_profile_env_step_s
```

Additional verification:

```bash
python -m py_compile rsl_rl/rsl_rl/runners/on_policy_runner.py utils/rl_nav_policy/env.py scripts/experiments/evaluate_rl_nav_policy.py scripts/experiments/train_rl_nav_policy.py
pytest -q utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 7 passed.

Follow-up fix:

- Fixed `KeyError` when `ep_infos` contains both sparse terminal dictionaries and dense step dictionaries.
- `OnPolicyRunner.log()` now iterates over the union of keys and skips dictionaries that do not contain the current key.
- `ActorCriticRLNav` import check passed.
- Hydra config composition check printed `ActorCriticRLNav False`.

### Uncertainty-Scaled Collision Penalty

`RLNavPolicyEnv` now supports scaling the physical collision penalty by local Neural Map confidence.

New reward config:

```yaml
rl_nav:
  reward:
    collision_uncertainty_scale:
      enabled: false
      scope: crop_mean
      min_scale: 0.75
      max_scale: 1.5
```

Behavior:

- The feature is disabled by default, preserving previous reward behavior.
- When enabled, the wrapper computes mean confidence from the current robot-centered uncertainty crop.
- Confidence uses the same log-scaled uncertainty representation as the policy observation.
- Physical collision penalty is scaled within `[min_scale, max_scale]`.
- Low uncertainty / high confidence produces a larger collision penalty.
- High uncertainty / low confidence produces a smaller but still bounded collision penalty.
- Map-out-of-bounds collision penalty remains fixed and is not reduced by uncertainty.

The first supported scope is `crop_mean`, matching the current 8x8 local crop semantics.

Additional verification:

```bash
python -m py_compile utils/rl_nav_policy/observation.py utils/rl_nav_policy/env.py utils/rl_nav_policy/__init__.py utils/rl_nav_policy/test_reward.py
pytest -q utils/rl_nav_policy/test_reward.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_terrain_metrics.py utils/rl_nav_policy/test_terrain_resampling.py utils/rl_nav_policy/test_goal_sampling.py utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_model.py
python - <<'PY'
import hydra
from omegaconf import OmegaConf
hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1')
cfg = hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test'])
OmegaConf.resolve(cfg)
print(cfg.rl_nav.reward.collision_uncertainty_scale.enabled, cfg.rl_nav.reward.collision_uncertainty_scale.scope, cfg.rl_nav.reward.collision_uncertainty_scale.min_scale, cfg.rl_nav.reward.collision_uncertainty_scale.max_scale)
PY
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 14 passed.
- Hydra config composition printed `False crop_mean 0.75 1.5`.

## Notes

- End-to-end IsaacGym training was not run during this change.
- Existing diffusion-related workspace modifications were left untouched.
- This is the Stage 1 architecture: local robot-centered crop plus goal/state condition. Coarse global map or goal-centered crop branches can be added later if long-range obstacle avoidance needs more global context.

## WandB Logging Update

The PPO training flow now supports optional WandB logging through the standard `OnPolicyRunner`.

Updated files:

- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - Added `runner.logger` support:
    - `tensorboard` keeps the existing default behavior.
    - `wandb` uses `rsl_rl.utils.wandb_utils.WandbSummaryWriter`.
    - `none` disables scalar writer logging while keeping terminal output and checkpoints.
  - Logs runner, algorithm, policy, and env config to WandB when available.
  - Saves checkpoints to WandB when using the WandB writer.
  - Calls `wandb.finish()` through `writer.stop()` at the end of training.
- `rsl_rl/rsl_rl/utils/wandb_utils.py`
  - Added optional config fields:
    - `wandb_run_name`
    - `wandb_entity`
    - `wandb_mode`
  - Falls back to the log directory name when `wandb_run_name` is not set.
- `scripts/experiments/train_rl_nav_policy.py`
  - Added CLI overrides:
    - `--logger tensorboard|wandb|none`
    - `--wandb_project`
    - `--wandb_mode`
- `scripts/experiments/train_residual_policy.py`
  - Added the same logger CLI overrides for residual PPO training.
- `scripts/experiments/configs/rl_nav_training.yaml`
  - Added default runner logging fields:

```yaml
logger: tensorboard
wandb_project: go2_neural_map_rl_nav
wandb_run_name: ${run_timestamp}
wandb_entity: null
wandb_mode: null
```

- `scripts/experiments/configs/residual_training.yaml`
  - Added equivalent defaults for residual training.

Example WandB run:

```bash
python scripts/experiments/train_rl_nav_policy.py \
  --config_name rl_nav_training \
  --logger wandb \
  --wandb_project go2_neural_map_rl_nav
```

Offline WandB run:

```bash
python scripts/experiments/train_rl_nav_policy.py \
  --config_name rl_nav_training \
  --logger wandb \
  --wandb_mode offline
```

Additional verification:

```bash
python -m py_compile rsl_rl/rsl_rl/runners/on_policy_runner.py rsl_rl/rsl_rl/utils/wandb_utils.py scripts/experiments/train_rl_nav_policy.py scripts/experiments/train_residual_policy.py
python -c "import hydra; from omegaconf import OmegaConf; hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1'); cfg=hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test']); OmegaConf.resolve(cfg); print(cfg.ppo.runner.logger, cfg.ppo.runner.wandb_project, cfg.ppo.runner.wandb_run_name)"
python -c "import hydra; from omegaconf import OmegaConf; hydra.initialize(config_path='scripts/experiments/configs', version_base='1.1'); cfg=hydra.compose(config_name='rl_nav_training', overrides=['+run_timestamp=test','ppo.runner.logger=wandb','ppo.runner.wandb_project=my_proj','ppo.runner.wandb_mode=offline']); OmegaConf.resolve(cfg); print(cfg.ppo.runner.logger, cfg.ppo.runner.wandb_project, cfg.ppo.runner.wandb_mode)"
```

Results:

- Python compile checks passed.
- Default config printed `tensorboard go2_neural_map_rl_nav test`.
- WandB override config printed `wandb my_proj offline`.

### WandB Float-Step Fix

WandB requires `step` to be an integer. The original runner logs TensorBoard time-axis metrics with `self.tot_time` as the global step:

```python
self.writer.add_scalar("Train/mean_reward/time", value, self.tot_time)
```

This crashed WandB with:

```text
TypeError: ... has type float, but expected one of: int
```

`WandbSummaryWriter.add_scalar()` now normalizes steps:

- integer `global_step` values are passed to WandB and remembered as the latest training iteration step.
- float `global_step` values are not passed directly as WandB steps.
- float-step metrics are logged at the latest integer step and include an additional `global_time` scalar.

Additional verification:

```bash
python -m py_compile rsl_rl/rsl_rl/utils/wandb_utils.py rsl_rl/rsl_rl/runners/on_policy_runner.py
pytest -q utils/rl_nav_policy/test_command.py utils/rl_nav_policy/test_observation.py utils/rl_nav_policy/test_model.py
```

Results:

- Python compile checks passed.
- `utils/rl_nav_policy` tests: 4 passed.
