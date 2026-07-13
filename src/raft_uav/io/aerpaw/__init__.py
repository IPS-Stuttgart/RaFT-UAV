"""Compatibility wrapper preserving truth-gated radar sequence boundaries.

The maintained implementation lives in the sibling ``aerpaw.py`` module. This
package preserves the public import path while ensuring that truth-gated radar
selection never matches a candidate against another sequence's truth trajectory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "aerpaw.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.io._aerpaw_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load AERPAW implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _truth_gate_mask(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> np.ndarray:
    """Return a truth-gate mask for rows already scoped to one sequence."""

    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    query_times = pd.to_numeric(radar["time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    if query_times.size == 0 or not bool(np.any(np.isfinite(truth_times))):
        return np.zeros(len(radar), dtype=bool)

    truth_indices = _IMPL._nearest_time_indices(truth_times, query_times)
    time_errors = np.abs(truth_times[truth_indices] - query_times)
    radar_xyz = radar[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    truth_xyz = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[
        truth_indices
    ]
    position_errors = np.linalg.norm(radar_xyz - truth_xyz, axis=1)
    return (time_errors <= float(truth_time_gate_s)) & (
        position_errors <= float(truth_gate_m)
    )


def _truth_gated_rows(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.DataFrame:
    """Apply truth gating without crossing optional sequence boundaries."""

    if "sequence_id" not in radar.columns or "sequence_id" not in truth.columns:
        keep = _truth_gate_mask(radar, truth, truth_gate_m, truth_time_gate_s)
        return radar.iloc[np.flatnonzero(keep)].copy()

    radar_sequence_ids = radar["sequence_id"].astype(str).to_numpy()
    truth_sequence_ids = truth["sequence_id"].astype(str).to_numpy()
    keep = np.zeros(len(radar), dtype=bool)
    for sequence_id in pd.unique(radar_sequence_ids):
        radar_positions = np.flatnonzero(radar_sequence_ids == sequence_id)
        truth_positions = np.flatnonzero(truth_sequence_ids == sequence_id)
        if truth_positions.size == 0:
            continue
        keep[radar_positions] = _truth_gate_mask(
            radar.iloc[radar_positions],
            truth.iloc[truth_positions],
            truth_gate_m,
            truth_time_gate_s,
        )
    return radar.iloc[np.flatnonzero(keep)].copy()


_IMPL._truth_gated_rows = _truth_gated_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_truth_gated_rows"] = _truth_gated_rows

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
