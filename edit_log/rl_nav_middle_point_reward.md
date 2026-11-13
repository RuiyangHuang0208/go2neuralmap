# RL Nav Middle Point Reward

## Background

This change adds optional terrain-local `middle_points` to terrain specs and exposes them to the RL navigation training wrapper as a per-task, one-shot reward target.

The intended semantics are:

- Path and middle point consistency is handled by manually designed terrain specs.
- The middle point is not added to the policy observation.
- A middle point reward can be triggered at most once per high-level task.
- Reaching the middle point does not terminate the task.

## Files Updated

- `utils/terrain_loader/terrain_loader.py`
  - Added `middle_points` to `TerrainSpec` and `TerrainInstance`.
  - Parses optional YAML key `middle_points`.
  - Missing `middle_points` is valid and represented as zero available middle points.
  - Added padded middle point tensors and count tensors.
  - Added:
    - `get_terrain_middle_points()`
    - `get_num_middle_points()`
    - `sample_middle_point_indices()`
    - `sample_middle_points()`
    - `has_middle_points()`
- `legged_gym/legged_gym/envs/go2/go2_rough_env.py`
  - Exposes terrain middle point sampling through:
    - `sample_middle_points(env_ids)`
    - `has_middle_points(env_ids)`
- `utils/rl_nav_policy/env.py`
  - Added reward config fields:
    - `middle_point_tolerance: float = 0.35`
    - `middle_point_reward: float = 0.0`
  - Added per-env buffers:
    - `middle_xy`
    - `middle_valid`
    - `middle_reward_claimed`
  - Samples one middle point per task from the current terrain when available.
  - Resets `middle_reward_claimed` on every `_reset_task(...)`.
  - Adds `middle_point_reward` once when the robot enters `middle_point_tolerance`.
  - Adds step/episode logging:
    - `step_reached_middle_rate`
    - `step_middle_valid_rate`
    - `step_middle_distance`
    - `middle_reached_rate`
- `utils/terrain_loader/test_terrain_loader.py`
  - Added tests for optional middle point parsing, padding, counts, valid masks, and sampling.
- `utils/rl_nav_policy/test_reward.py`
  - Added tests for middle point reward, one-shot claimed state, and per-task reset behavior.

## Terrain YAML Format

`middle_points` is optional and uses the same terrain-local `[x, y, z]` format as `starts` and `goals`:

```yaml
middle_points:
  - [4.0, 4.0, 0.0]
  - [5.0, 3.0, 0.0]
```

If a terrain has no `middle_points`, `middle_valid=False` for tasks sampled on that terrain and no middle point reward is emitted.

## Reward Config

The reward is disabled by default because `middle_point_reward` defaults to `0.0`.

Example:

```yaml
rl_nav:
  reward:
    middle_point_reward: 3.0
    middle_point_tolerance: 0.35
```

## Config Check

Checked the composed config:

```text
config_name=rl_nav_training
+exp=training/rl_nav/stage3_v3_nm_enhancement
```

The config currently has an invalid nested reward block:

```yaml
rl_nav:
  reward:
    reward:
      middle_point_reward: 3.0
      middle_point_tolerance: 0.35
```

This causes:

```text
ConfigKeyError: Key 'reward' not in 'RLNavRewardConfig'
full_key: reward.reward
```

The fields should be moved directly under `rl_nav.reward`.

The selected terrain file is:

```text
resources/terrains/configs/rl_nav/nm_enhancement.yaml
```

It contains 18 terrain entries, 12 of which define `middle_points`. The other 6 entries have no middle point and will not trigger the middle point reward.

## Entropy Schedule Check

The same config sets:

```yaml
entropy_coef: 0.01
entropy_coef_schedule: linear
entropy_coef_final: 0.001
entropy_coef_schedule_steps: 1000
```

`rsl_rl/rsl_rl/algorithms/ppo.py` supports this schedule and `OnPolicyRunner.learn(...)` passes the current learning iteration into `self.alg.update(it)`.

Expected entropy coefficient values:

```text
iter 0    -> 0.010000
iter 500  -> 0.005500
iter 1000 -> 0.001000
iter 1500 -> 0.001000
```

`schedule: adaptive` is the separate KL-based learning-rate schedule and does not conflict with the entropy coefficient schedule.

## Validation

Ran:

```bash
pytest -q utils/terrain_loader/test_terrain_loader.py
pytest -q utils/rl_nav_policy
python -m py_compile utils/terrain_loader/terrain_loader.py utils/rl_nav_policy/env.py legged_gym/legged_gym/envs/go2/go2_rough_env.py
```

Results:

```text
utils/terrain_loader/test_terrain_loader.py: 9 passed
utils/rl_nav_policy: 27 passed
py_compile: passed
```
