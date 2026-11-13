#  Overview

Here I want to implement a new velocity_command_generator that uses diffusion policy under @scripts/experiments/modules/vel_command_generator

# Requirements
- Follow current implementation convention
- This module 
    - gets relevant context from @scripts/experiments/starter.py, and then computes and concatenate observation for the diffusion policy module
    - computes and convert results into velocity command sequence
- Here we use horizon receding method. The horizon can be configured, by default here is 2
- Time interval between velocity commands within the sequence is 0.5s (as set in training settings), thus it should be converted into steps according to main loop frequency. By default
    - The main loop executes at 50 Hz
    - Physic step runs at 200 Hz, i.e. decimation is 4

# Supplementary Information
- Observation structure can be seen 
    - at dataset preparation process defined in @scripts/experiments/augment_dataset.py
    - at metadata of existing dataset @go2_smoke/metadata.json
    - at training pipeline defined in @utils/diffusion_policy

# Meta Instructions
- You need to ask about anything you are not clear
- If you have better choice or option, just suggest it
