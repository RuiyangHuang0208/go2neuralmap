# RLNavV3 Dual-Scale Global Map Variant

## Summary

Adds `ActorCriticRLNavV3`, a dual-scale RL navigation policy variant. V3 keeps the RLNavV2 local conditioned-query attention structure, but adds a pooled global NeuralMap branch that conditions query generation.

## Design

- Local map branch:
  - Uses the existing high-resolution local crop tokens.
  - Local tokens remain the only key/value source for cross-attention.

- Global map branch:
  - Reads the full NeuralMap.
  - Pools it to `global_pool_size x global_pool_size`, defaulting to `12x12`.
  - Uses confidence-weighted latent pooling from scaled uncertainty.
  - Recomputes geometry tokens from pooled cell centers.
  - Encodes pooled global tokens into a single global embedding.

- Query generation:
  - V3 query is generated from `[condition_feature, local_embedding, global_embedding]`.
  - Actor and critic each own an independent dual-scale encoder.

## Files Changed

- `utils/rl_nav_policy/observation.py`
  - Adds `include_global_map` and `global_pool_size` to `RLNavObservationConfig`.
  - Keeps the old V1/V2 observation layout when `include_global_map=false`.
  - Adds pooled global map token generation for V3.

- `utils/rl_nav_policy/model.py`
  - Adds `RLNavDualScaleConditionedQueryMapEncoder`.

- `rsl_rl/rsl_rl/modules/actor_critic_rl_nav_v3.py`
  - Adds `ActorCriticRLNavV3`.

- `rsl_rl/rsl_rl/modules/__init__.py`
  - Exports `ActorCriticRLNavV3`.

- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - Imports `ActorCriticRLNavV3` for the existing `eval(policy_class_name)` path.

- `scripts/experiments/train_rl_nav_policy.py`
  - Writes `local_crop_size`, `global_pool_size`, `token_dim`, and `condition_dim` into PPO policy config from the constructed env.

- `scripts/experiments/configs/rl_nav_training_v3.yaml`
  - Adds a standalone V3 training config.

- `utils/rl_nav_policy/test_observation.py`
  - Adds dual-scale observation layout coverage.

- `utils/rl_nav_policy/test_model.py`
  - Adds V3 forward and attention-grid shape coverage.

## Training

Run:

```bash
python scripts/experiments/train_rl_nav_policy.py --config_name rl_nav_training_v3
```

The V3 config writes runs to:

```text
logs/rl_nav_policy/${run_timestamp}_v3
```

## Notes

- With the default map configuration (`48x48` at `0.25m` resolution), `global_pool_size=12` pools each global token over a `4x4` block, approximately `1.0m x 1.0m`.
- Observation tokens increase from `64` local tokens to `64 + 144 = 208` tokens, but attention still attends only over the `64` local tokens.
