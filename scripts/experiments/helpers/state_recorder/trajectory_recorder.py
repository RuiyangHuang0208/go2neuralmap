from .sub_recorder import SubRecorder
from loguru import logger
import torch
from copy import deepcopy

class TrajectoryRecorder(SubRecorder):

    def __init__(self, **kwargs):
        self.trajs = []
        self.grid_trajs = []
        self.logger = logger.bind(module_name="StateRecorder.TrajRec")

        self.curr_traj = []
        self.curr_grid_traj = []
        self.traj_costs = []

        self.last_grid_pos = None

        self.sampling_decimation = 10
        self.step_cnt = 0
        pass

    def handle(self, var_dict):
        vel_cmd_info = var_dict.get("vel_cmd_generator_info")
        if (vel_cmd_info is None) or (len(vel_cmd_info.keys()) == 0):
            self.logger.warning("Got no traj information")
            return
        reset_buf = var_dict.get("reset_buf")
        done = bool(reset_buf[0].item()) if isinstance(reset_buf, torch.Tensor) and reset_buf.numel() > 0 else False
        traj_cost = vel_cmd_info.get("traj_cost")

        if self.step_cnt % self.sampling_decimation == 0:
            curr_pos = vel_cmd_info.get("current_xy")
            if curr_pos is not None:
                self.step_cnt = 0
                self.curr_traj.append(curr_pos)

        curr_grid_pos = vel_cmd_info.get("current_grid_xy")
        if curr_grid_pos is not None:
            if self.last_grid_pos is None:
                self.last_grid_pos = curr_grid_pos
            if self.last_grid_pos != curr_grid_pos:
                self.curr_grid_traj.append(deepcopy(self.last_grid_pos))
            self.last_grid_pos = curr_grid_pos

        if bool(vel_cmd_info.get("reached_goal_flag", False)) or done:
            if len(self.curr_traj) != 0 and len(self.curr_grid_traj) != 0:
                self.trajs.append(torch.stack(deepcopy(self.curr_traj), dim=0))
                self.grid_trajs.append(torch.tensor(deepcopy(self.curr_grid_traj)))
                self.traj_costs.append(traj_cost)
            self.curr_traj = []
            self.curr_grid_traj = []
            self.last_grid_pos = None

        self.step_cnt += 1

    def save(self, save_dir):
        trajs = list(self.trajs)
        grid_trajs = list(self.grid_trajs)
        traj_costs = list(self.traj_costs)

        if len(self.curr_traj) != 0:
            trajs.append(torch.stack(deepcopy(self.curr_traj), dim=0))
        if len(self.curr_grid_traj) != 0:
            grid_trajs.append(torch.tensor(deepcopy(self.curr_grid_traj)))
        if len(trajs) > len(traj_costs):
            traj_costs.extend([None] * (len(trajs) - len(traj_costs)))

        torch.save(
            {
                "trajs": trajs,
                "grid_trajs": grid_trajs,
                "traj_costs": traj_costs,
            },
            f"{save_dir}/trajectories.pt",
        )
        pass
