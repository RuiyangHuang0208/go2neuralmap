try:
    from .go2_rough_config import GO2RoughCfg, GO2RoughCfgPPO
except ImportError:
    from legged_gym.envs.go2.go2_rough_config import GO2RoughCfg, GO2RoughCfgPPO
    print("Import module from absolute path for debugging")
from dataclasses import dataclass


@dataclass
class GO2NMCfg(GO2RoughCfg):
    # ['base', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'Head_upper', 'Head_lower', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot', 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot']
    class cost:
        key_bodies = ["Head", "thigh", "calf"]
        visualize_collision_contacts = False
        collision_contact_threshold = 5.0
        collision_contact_viz_marker_radius = 0.05

        body_direction_group = {
            "front": [ "Head"     , "FL_thigh" , "FR_thigh"] ,
            "left":  [ "FL_thigh"  , "RL_thigh" , ]           ,
            "right": [ "FR_thigh" , "RR_thigh" , ]           ,
            "rear":  [ "RL_thigh"  , "RR_thigh" , ]           ,
        }

        class scales:
            power = 1.0
            collision = 1.0
            termination = 1.0
        
    pass

    class asset(GO2RoughCfg.asset):
        robot_mass = 15 # [kg]

class GO2NMCfgPPO(GO2RoughCfgPPO):
    pass
