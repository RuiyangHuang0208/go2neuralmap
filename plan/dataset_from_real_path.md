# Instruction

## Goal
I want to implement a new dataset pipeline to collect position, neural map, velocity sequence from real rollout situation

## Context
This dataset collection pipeline is used for a Diffusion Policy training pipeline. The training is aiming to get a velocity command generator, which takes neural map as input.

The old dataset collector collects a rough dataset, which will be augmented after collection. These data will be used to pre-training.

Now a new dataset collector will be created, in order to directly collect ground truth data from rollout, rather than from generation/augmentation. Data from new data collector will be used for refinement.

## Scope
List what should be changed.

- In scope:
    - Add a variation of dataset collector
    - Refactor current dataset collector configuration method, use strategy mode
- Out of scope:
    - Do not change behavior and logic of older dataset collector

## Target Files / Areas
Mention known files, modules, or packages to inspect or modify.

- `scripts/experiments/starter.py`
- `scripts/experiments/configs`
- `scripts/experiments/helpers`

## Requirements
State concrete behavior and constraints.

1. Follow the current data collector implementation convention
2. Dataset structure should be like metadata in `go2_smoke/metadata.json`
3. Velocity and heading command should come from real robot states:
    - Vel is in local frame, heading is in global frame
    - For e.g. I want to collect vel cmd sequence of length equals 10 with dt=0.5s, then every 0.5s samples information.

## Clarified Design

- The velocity/heading sequence stored in the dataset is not copied from `vel_cmd_generator` output or `env.commands`.
- Each sequence element is collected from the robot state at rollout time:
    - local velocity: robot body-frame planar velocity, e.g. `env.base_lin_vel[env_id, :2]`
    - global heading: yaw angle decoded from root orientation, e.g. `env.root_states[env_id, 3:7]`
- The collector should use strategy mode so the old collector remains behaviorally unchanged and a new real-path collector can be selected from config.
- The real-path collector should emit augmented-dataset-compatible fields where possible:
    - `velocity_commands`: `[velocity_horizon, 3]` with `[local_vx, local_vy, global_heading]`
    - `velocity_command_mask`: valid future samples in the horizon
    - `start_xy`: robot position at anchor step
    - `goal_xy`: robot position at the last valid future sample
    - path/waypoint fields can represent the observed future rollout path for schema compatibility
- If a rollout terminates before the full future horizon, keep the sample with a partial mask unless configured to drop incomplete horizons.
