"""Periodic raw snapshots of a directional uncertainty map and base trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger

from .sub_recorder import SubRecorder


class UncertaintyFootprintRecorder(SubRecorder):
    """Persist aged directional uncertainty maps with the base XY trajectory.

    The recorder receives one point per control step, but writes snapshots at a
    separate wall-clock-in-simulation interval.  ``uncertainty_map_vals`` must
    use the channel-first layout ``[4, x_cells, y_cells]`` with channels
    ``[+x, -x, +y, -y]``.
    """

    CHANNEL_NAMES = ("pos_x", "neg_x", "pos_y", "neg_y")

    def __init__(self, *, log_dir: str = "", snapshot_interval_seconds: float = 60.0, **kwargs):
        if snapshot_interval_seconds <= 0.0:
            raise ValueError("snapshot_interval_seconds must be positive")

        self.snapshot_interval_seconds = float(snapshot_interval_seconds)
        self.output_dir = Path(log_dir) / "state_recorder" / "uncertainty_footprint"
        self.logger = logger.bind(module_name="StateRecorder.UncertaintyFootprint")
        self.footprint_xy = []
        self.elapsed_seconds = 0.0
        self.elapsed_since_snapshot = 0.0
        self.step = -1
        self.last_snapshot_step: Optional[int] = None
        self.latest_uncertainty: Optional[torch.Tensor] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._warned_missing_uncertainty = False

    def handle(self, var_dict):
        current_xy = var_dict.get("current_xy")
        if isinstance(current_xy, torch.Tensor) and current_xy.numel() >= 2:
            self.footprint_xy.append(current_xy.detach().cpu().to(dtype=torch.float32).flatten()[:2].clone())

        self.step = int(var_dict.get("step", self.step + 1))
        dt = self._step_time(var_dict.get("dt"))
        self.elapsed_seconds += dt
        self.elapsed_since_snapshot += dt

        uncertainty = var_dict.get("uncertainty_map_vals")
        if uncertainty is None:
            if not self._warned_missing_uncertainty:
                self.logger.warning(
                    "Directional uncertainty map unavailable; uncertainty/footprint snapshots will not be written."
                )
                self._warned_missing_uncertainty = True
            return
        if not isinstance(uncertainty, torch.Tensor) or uncertainty.ndim != 3 or uncertainty.shape[0] != 4:
            raise ValueError("uncertainty_map_vals must have shape [4, x_cells, y_cells]")

        self.latest_uncertainty = uncertainty.detach().cpu().to(dtype=torch.float32).clone()
        metadata = var_dict.get("uncertainty_map_metadata")
        if isinstance(metadata, dict):
            self.metadata = dict(metadata)

        if self.elapsed_since_snapshot >= self.snapshot_interval_seconds:
            self._write_snapshot()

    def save(self, save_dir):
        if self.latest_uncertainty is not None and self.last_snapshot_step != self.step:
            self._write_snapshot()

    def _write_snapshot(self):
        if self.latest_uncertainty is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        footprint = (
            torch.stack(self.footprint_xy, dim=0)
            if self.footprint_xy
            else torch.empty((0, 2), dtype=torch.float32)
        )
        torch.save(
            {
                "step": self.step,
                "elapsed_seconds": self.elapsed_seconds,
                "uncertainty": self.latest_uncertainty,
                "footprint_xy": footprint,
            },
            self.output_dir / f"snapshot_step_{self.step:06d}.pt",
        )
        if self.metadata is not None:
            torch.save(
                {
                    **self.metadata,
                    "channel_names": self.CHANNEL_NAMES,
                    "uncertainty_semantics": "aged",
                    "snapshot_interval_seconds": self.snapshot_interval_seconds,
                },
                self.output_dir / "metadata.pt",
            )
        self.last_snapshot_step = self.step
        self.elapsed_since_snapshot = 0.0

    @staticmethod
    def _step_time(value: Any) -> float:
        try:
            dt = float(value)
        except (TypeError, ValueError):
            return 0.0
        return dt if dt > 0.0 else 0.0
