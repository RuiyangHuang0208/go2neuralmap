from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from loguru import logger
from typing import Any, Dict, List, Optional

from .sub_recorder import SubPlotter, SubRecorder
from .subrecorder_registry import SubPlotterRegistry, SubRecorderRegistry


@dataclass
class StateRecorderConfig():
    interactive_mode: bool = True
    sampling_interval: int = 5  # Every 1 / 50 * 5 = 0.1s, add new points to plots
    replot_decimation: int = 5
    sub_plotters: List[str] = field(default_factory=lambda: [
        "foot_print",
    ])
    sub_recorders: List[str] = field(default_factory=lambda: [
        "trajectory_recorder"
    ])
    sub_recorder_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    log_dir: str = ""
    pass


class BaseStateRecorder(ABC):
    @abstractmethod
    def collect_vars(self, var_dict: dict):
        pass

    def handle(self):
        pass

    def save(self):
        pass


class NullStateRecorder(BaseStateRecorder):
    def __init__(self, *args, **kwargs):
        pass

    def collect_vars(self, var_dict):
        return

    def handle(self):
        return

    def save(self):
        return


class StateRecorder(BaseStateRecorder):
    def __init__(self, cfg: StateRecorderConfig = StateRecorderConfig()):
        self.cfg = cfg
        self.var_dict = {}
        self.logger = logger.bind(module_name="StateRecorder")

        if self.cfg.interactive_mode:
            plt.ion()

        self.fig: Optional[Figure] = None
        self.sub_plotters = self._allocate_ax()
        self.sub_recorders = [
            SubRecorderRegistry[name](
                log_dir=self.cfg.log_dir,
                **self.cfg.sub_recorder_configs.get(name, {}),
            )
            for name in self.cfg.sub_recorders
        ]

        self.sampling_interval = self.cfg.sampling_interval
        self.replot_decimation = self.cfg.replot_decimation

        self.re_plot_cnt = 0
        self.sample_cnt = 0

        pass

    def _allocate_ax(self) -> List:
        from math import ceil
        num_sub_plots = len(self.cfg.sub_plotters)
        n_rows = ceil(num_sub_plots / 2.)
        if n_rows == 0:
            return []

        self.fig: Figure
        self.fig, axes = plt.subplots(n_rows, 2)
        self.fig.set_size_inches(14, n_rows * 6)

        axes_flatten = axes.reshape(-1)

        sub_plotters = []
        for idx, cls_name in enumerate(self.cfg.sub_plotters):
            plotter = SubPlotterRegistry[cls_name](axes_flatten[idx])
            sub_plotters.append(plotter)

        return sub_plotters

    def _replot(self):
        if self.fig is None:
            return
        self.re_plot_cnt += 1
        if self.re_plot_cnt % self.replot_decimation == 0:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.re_plot_cnt = 0

    def handle(self):
        self.sample_cnt += 1
        if self.sample_cnt % self.sampling_interval == 0:
            draw_flg = False
            for sub_plotter in self.sub_plotters:
                sub_plotter: SubPlotter
                sub_plotter.draw(self.var_dict)
                draw_flg = True
            if draw_flg:
                self.logger.debug("Plot drawn")
            self._replot()

        for sub_recorder in self.sub_recorders:
            sub_recorder.handle(self.var_dict)

    def collect_vars(self, var_dict):
        self.var_dict = var_dict
        pass

    def save(self):
        import os
        state_rec_dir = os.path.join(self.cfg.log_dir, "state_recorder")
        os.makedirs(state_rec_dir, exist_ok=True)
        if self.fig is not None:
            self.fig.savefig(f"{state_rec_dir}/states.png")
        for sub_plotter in self.sub_plotters:
            sub_plotter.save(state_rec_dir)

        for sub_recorder in self.sub_recorders:
            sub_recorder.save(state_rec_dir)
