# RL Nav V3 No-Attention Ablation

## Purpose

Add a V3 ablation policy that removes the condition-query cross-attention over local map tokens. The ablation keeps the V3 dual-scale observation layout and encoder output width, but replaces the attention-derived local feature with an independently projected mean-pooled local context.

## Model Change

Original V3 encoder output:

```python
[attended_local, local_embedding, global_embedding, condition_feature]
```

No-attention ablation output:

```python
[pooled_local_context, local_embedding, global_embedding, condition_feature]
```

`pooled_local_context` and `local_embedding` both start from `local_features.mean(dim=1)`, but they use separate MLPs. This avoids feeding two copies of the same local embedding to the policy head.

The no-attention encoder does not instantiate `torch.nn.MultiheadAttention` and does not use the conditioned query projection from the original V3 attention encoder.

## Files Added

- `rsl_rl/rsl_rl/modules/actor_critic_rl_nav_v3_no_attention.py`
- `scripts/experiments/configs/exp/training/rl_nav/stage0_v3_no_attention.yaml`
- `scripts/experiments/configs/exp/training/rl_nav/stage1_v3_no_attention.yaml`
- `scripts/experiments/configs/exp/training/rl_nav/stage2_v3_no_attention.yaml`
- `scripts/experiments/configs/exp/training/rl_nav/stage3_v3_no_attention.yaml`
- `edit_log/rl_nav_v3_no_attention_ablation.md`

## Files Modified

- `utils/rl_nav_policy/model.py`
  - Added `RLNavDualScalePooledMapEncoder`.
- `rsl_rl/rsl_rl/modules/__init__.py`
  - Exported `ActorCriticRLNavV3NoAttention`.
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - Imported `ActorCriticRLNavV3NoAttention` so `policy_class_name` resolves in the runner.
- `utils/rl_nav_policy/test_model.py`
  - Added V3 no-attention forward shape, uniform grid, and no-MHA tests.

## Training Configs

Added one no-attention config for each RL nav stage:

- `stage0_v3_no_attention.yaml`
- `stage1_v3_no_attention.yaml`
- `stage2_v3_no_attention.yaml`
- `stage3_v3_no_attention.yaml`

All configs use:

```yaml
policy_class_name: ActorCriticRLNavV3NoAttention
```

The policy config keeps the V3 dual-scale map layout:

```yaml
local_crop_size: ${rl_nav.observation.crop_size}
global_pool_size: ${rl_nav.observation.global_pool_size}
local_context_hidden_dims: [256]
local_crop_encoder_hidden_dims: [256]
global_encoder_hidden_dims: [256]
```

## Stage 3 Notes

`stage3_v3_no_attention.yaml` follows `stage3_condition_only.yaml` for the requested refinement settings:

```yaml
env:
  terrain:
    num_rows: 12
    num_cols: 10
    terrains_yaml: resources/terrains/configs/rl_nav/nm_enhancement.yaml
```

Entropy schedule is also matched:

```yaml
entropy_coef: 0.01
entropy_coef_schedule: linear
entropy_coef_final: 0.001
entropy_coef_schedule_steps: 1000
```

Repeated collision penalty is enabled with the same values as `stage3_condition_only.yaml`. `collision_uncertainty_scale.enabled` is set to `false` to match the condition-only stage3 reference.

## Compatibility Note

`ActorCriticRLNavV3NoAttention.get_actor_attention_grid()` returns a uniform local pooling grid for compatibility with existing visualization/test calls. It is not a real attention map.

## Verification

Passed syntax compilation:

```bash
python3 -m py_compile utils/rl_nav_policy/model.py rsl_rl/rsl_rl/modules/actor_critic_rl_nav_v3_no_attention.py rsl_rl/rsl_rl/modules/__init__.py rsl_rl/rsl_rl/runners/on_policy_runner.py utils/rl_nav_policy/test_model.py
```

Checked stage3 config fields with `rg` and confirmed:

- `num_rows: 12`
- `terrains_yaml: resources/terrains/configs/rl_nav/nm_enhancement.yaml`
- `policy_class_name: ActorCriticRLNavV3NoAttention`
- entropy schedule `0.01 -> 0.001` over `1000` steps

After the Python environment was updated, the targeted and package tests passed:

```bash
python3 -m pytest -q utils/rl_nav_policy/test_model.py
python3 -m pytest -q utils/rl_nav_policy
```

Results:

```text
10 passed in 2.02s
34 passed in 2.18s
```

The shell printed a `libtinfo.so.6` version warning from the active conda environment, but pytest completed successfully.
