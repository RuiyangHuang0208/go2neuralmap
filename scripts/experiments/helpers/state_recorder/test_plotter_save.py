import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from scripts.experiments.helpers.state_recorder.cost_map_plotter import CostMap
from scripts.experiments.helpers.state_recorder.foot_print_plotter import FootPrint


def test_foot_print_save_without_draw_writes_empty_tensor(tmp_path):
    _, ax = plt.subplots()
    plotter = FootPrint(ax)

    plotter.save(tmp_path)

    saved = torch.load(tmp_path / "foot_print.pt", weights_only=True)
    assert torch.equal(saved["foot_print"], torch.empty((0, 2), dtype=torch.float32))


def test_foot_print_save_after_first_draw_writes_first_point(tmp_path):
    _, ax = plt.subplots()
    plotter = FootPrint(ax)

    plotter.draw({"current_xy": torch.tensor([1.0, 2.0])})
    plotter.save(tmp_path)

    saved = torch.load(tmp_path / "foot_print.pt", weights_only=True)
    assert torch.equal(saved["foot_print"], torch.tensor([[1.0, 2.0]]))


def test_cost_map_save_without_draw_writes_empty_tensor(tmp_path):
    _, ax = plt.subplots()
    plotter = CostMap(ax)

    plotter.save(tmp_path)

    saved = torch.load(tmp_path / "cost_map_vals.pt", weights_only=True)
    assert torch.equal(saved["cost_map_vals"], torch.empty((0, 0, 0), dtype=torch.float32))
