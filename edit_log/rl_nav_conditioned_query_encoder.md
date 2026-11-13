# RL Navigation Conditioned Query Map Encoder

## Summary

This change adds an isolated RL navigation policy network variation:

```text
ActorCriticRLNavV2
```

The PPO interface, observation layout, actor/critic separation, action shape, and attention visualization API are kept compatible with the original `ActorCriticRLNav`.

The main network change is inside the map encoder. Instead of building the cross-attention query from only the condition feature, the new encoder first pools projected map token features into a crop-level embedding, then builds the query from both:

```text
condition_feature + crop_embedding
```

This lets the attention query depend on the task condition and the overall cropped Neural Map context.

## Files Added

- `rsl_rl/rsl_rl/modules/actor_critic_rl_nav_v2.py`
  - Defines `ActorCriticRLNavV2`.
  - Uses separate actor and critic encoders.
  - Preserves the same PPO-facing methods as `ActorCriticRLNav`.
  - Preserves `get_actor_attention_grid()` for attention visualization.

## Files Updated

- `utils/rl_nav_policy/model.py`
  - Adds `RLNavConditionedQueryMapEncoder`.
  - Projects per-cell tokens into `token_features`.
  - Pools `token_features.mean(dim=1)` into a crop-level embedding through `crop_proj`.
  - Builds the attention query from `concat(condition_feature, crop_embedding)` through `query_proj`.
  - Keeps encoder output dimension as `embed_dim * 3`:

    ```text
    concat(attended_feature, crop_embedding, condition_feature)
    ```

- `rsl_rl/rsl_rl/modules/__init__.py`
  - Exports `ActorCriticRLNavV2`.

- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - Imports `ActorCriticRLNavV2` so the existing `eval(policy_class_name)` path can instantiate it.

- `utils/rl_nav_policy/test_model.py`
  - Adds forward shape coverage for `ActorCriticRLNavV2`.
  - Adds attention grid shape and normalization coverage for `ActorCriticRLNavV2`.

## Encoder Structure

The original encoder path is:

```text
tokens -> token_proj -> token_features
condition -> condition_proj -> condition_feature
query = condition_feature
cross_attention(query, token_features, token_features)
global_feature = global_proj(mean(token_features))
output = concat(attended, global_feature, condition_feature)
```

The new V2 encoder path is:

```text
tokens -> token_proj -> token_features
condition -> condition_proj -> condition_feature
crop_embedding = crop_proj(mean(token_features))
query = query_proj(concat(condition_feature, crop_embedding))
cross_attention(query, token_features, token_features)
output = concat(attended, crop_embedding, condition_feature)
```

The output dimension remains unchanged at `embed_dim * 3`, so the actor and critic MLP heads can keep the same shape assumptions.

## Training Configuration

To use this variation, set:

```yaml
ppo:
  runner:
    policy_class_name: ActorCriticRLNavV2
  policy:
    crop_encoder_hidden_dims: [256]
    query_hidden_dims: [256]
```

Other existing RL navigation policy fields remain valid, including:

```yaml
encoder_embed_dim: 128
encoder_heads: 4
encoder_hidden_dims: [256]
actor_hidden_dims: [256, 256]
critic_hidden_dims: [256, 256]
attention_dropout: 0.0
```

The training entry point still updates `crop_size`, `token_dim`, and `condition_dim` from the constructed `RLNavPolicyEnv` before creating the PPO runner.

## Verification

Command run:

```bash
pytest -q utils/rl_nav_policy/test_model.py
```

Result:

```text
4 passed
```

## Notes

- The original `ActorCriticRLNav` and `RLNavMapEncoder` are preserved.
- The new policy is isolated behind `policy_class_name: ActorCriticRLNavV2`.
- Actor and critic use independent instances of the new encoder, matching the original policy structure.
- Existing unrelated untracked workspace items were not modified.
