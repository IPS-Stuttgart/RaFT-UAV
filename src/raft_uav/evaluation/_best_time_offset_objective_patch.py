"""Use metric-aware and deterministic time-offset selection."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

_radar_oracle = import_module("raft_uav.evaluation.radar_oracle_diagnostics")
_MAXIMIZE_METRICS = frozenset({"count", "coverage"})


def best_time_offset(
    sweep: pd.DataFrame,
    *,
    metric: str = "mean_3d_error_m",
) -> float | None:
    """Return the best finite offset using the selected metric's direction.

    Error metrics are minimized, while candidate count and coverage are
    maximized. Metric ties prefer the smallest absolute correction and then the
    numerically smaller signed offset, making the result independent of row
    order.
    """

    if (
        sweep.empty
        or metric not in sweep.columns
        or "time_offset_s" not in sweep.columns
    ):
        return None

    values = pd.to_numeric(sweep[metric], errors="coerce").to_numpy(dtype=float)
    offsets = pd.to_numeric(
        sweep["time_offset_s"],
        errors="coerce",
    ).to_numpy(dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values) & np.isfinite(offsets))
    if finite_indices.size == 0:
        return None

    finite_values = values[finite_indices]
    best_value = float(
        np.max(finite_values)
        if metric in _MAXIMIZE_METRICS
        else np.min(finite_values)
    )
    tied_indices = finite_indices[finite_values == best_value]
    tied_offsets = offsets[tied_indices]
    tie_order = np.lexsort((tied_offsets, np.abs(tied_offsets)))
    return float(tied_offsets[int(tie_order[0])])


def install() -> None:
    """Install the selector on public and maintained implementation paths."""

    if getattr(_radar_oracle, "_best_time_offset_objective_patch_applied", False):
        return
    _radar_oracle.best_time_offset = best_time_offset
    implementation = getattr(_radar_oracle, "_IMPL", None)
    if implementation is not None:
        implementation.best_time_offset = best_time_offset
    _radar_oracle._best_time_offset_objective_patch_applied = True


install()
