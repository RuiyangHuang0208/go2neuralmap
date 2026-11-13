# Residual Policy Training Optimization

## Background

Training with `residual_training_1` produced a conservative residual policy that frequently cancelled the diffusion velocity command. The likely cause was the combination of high residual authority, sparse goal reward, and large collision penalty.

## Changes

- Added residual command authority controls in `utils/residual_policy/command_mixer.py`:
  - `residual_vel_clip`
  - `residual_heading_clip`
  - `mix_mode`
  - `min_velocity_scale`
  - `max_velocity_scale`
- Kept the old additive behavior as the default for backward compatibility.
- Added optional `scale_velocity` mixing mode for future ablations.
- Extended residual reward configuration in `utils/residual_policy/env.py`:
  - split velocity and heading residual penalties
  - anti-cancellation penalty
  - positive-only dense progress reward
  - per-step time penalty
- Added richer residual diagnostics to `extras["episode"]`:
  - `base_speed`
  - `mixed_speed`
  - `speed_ratio`
  - `vel_residual_norm`
  - `heading_residual_abs`
  - `anti_cancellation`
  - `positive_progress`
- Updated residual inference and command-generator construction so mixer clipping/scaling options can be reused outside training.
- Added `scripts/experiments/configs/exp/training/residual_training_2.yaml`.

## New Training Configuration

`residual_training_2.yaml` keeps additive residuals but restricts them:

- `residual_vel_clip: 0.2`
- `residual_heading_clip: 0.3`
- `collision_penalty: 1.0`
- `variant: shaped`
- `progress_positive_only: true`
- `progress_scale: 2.0`
- `anti_cancellation_penalty_scale: 0.2`
- `lam: 0.98`

This setup allows the residual policy to make local safety corrections while making habitual cancellation of the diffusion command costly.

## Notes

- Old configs without the new reward fields still use the legacy raw action penalty.
- `scale_velocity` is implemented for ablation, but `residual_training_2` uses the recommended `additive` mode.
- Deployment configs should include the same mixer clip values as the training config when evaluating a policy trained with `residual_training_2`.
