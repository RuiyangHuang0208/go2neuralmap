from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

class GO2RoughCfg( LeggedRobotCfg ):

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'
        measure_heights = False

        num_rows = 2
        num_cols = 2
        gap = 1.0
        terrains_yaml = str("/home/liwei/F/002Project/2025W/MasterThesis/go2_neural_map/resources/terrains/terrain.yaml")
        device = None
        static_friction = 0.8
        dynamic_friction = 0.7
        restitution = 0.1

    class env( LeggedRobotCfg.env ):
        # num_proprio_obs = 42#+187
        num_proprio_obs = 42#+187
        num_short_obs_history_length = 6
        num_long_obs_history_length = 100
        dim_long_history_latent=32
        randomize_reset_heading = False

        num_observations = 3 + num_proprio_obs +3 + 1 + num_long_obs_history_length*num_proprio_obs  + num_short_obs_history_length*num_proprio_obs
            # num_observations = num_proprio_obs + num_long_obs_history_length*num_proprio_obs  + num_short_obs_history_length*num_proprio_obs
        
        # use_ground_truth_velocity_for_obs = False # if true: use ground truth velocity in obs, else use estimated velocity

        include_estimated_velocity_in_obs = False
        
        num_privileged_obs = 3 + num_proprio_obs+187+45 #+ num_long_obs_history_length*num_proprio_obs
        # num_privileged_obs = num_proprio_obs+187+45 #+ num_long_obs_history_length*48
        obs_history_stride = 1
        
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
        # stiffness = {'joint': 20.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    # class commands( LeggedRobotCfg.commands ):
    #     curriculum = False
    #     max_curriculum = 1.
    #     num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
    #     resampling_time = 10. # time before command are changed[s]
    #     heading_command = True # if true: compute ang vel command from heading error
    #     class ranges:
    #         lin_vel_x = [-1.0, 1.0] # min max [m/s]
    #         lin_vel_y = [-1.0, 1.0]   # min max [m/s]
    #         ang_vel_yaw = [-1, 1]    # min max [rad/s]
    #         heading = [-3.14, 3.14]


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        # penalize_contacts_on = ["thigh", "calf","hip"]
        # penalize_contacts_on = ["thigh", "calf","base","Head"]
        penalize_contacts_on = ["thigh", "calf","Head"]
        # penalize_contacts_on = ["Head"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  


    # class eval( LeggedRobotCfg.eval ):
    #     eval_mode = False

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5 #5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
        # wandb_project="rough_go2_transformer_encoder"
        wandb_project="rough_go2_benchmark"


        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        # num_steps_per_env = 24 # per iteration
        max_iterations = 10000 # number of policy updates

  
