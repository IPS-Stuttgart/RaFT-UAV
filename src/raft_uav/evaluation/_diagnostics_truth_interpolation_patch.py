"""Align compact diagnostic errors with interpolated truth positions."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from . import metrics as _metrics

_diagnostics = import_module("raft_uav.evaluation.diagnostics")


def _position_error_frame(
    *,
    estimate_frame: pd.DataFrame,
    truth: pd.DataFrame,
    max_eval_time_delta_s: float | None,
) -> pd.DataFrame:
    """Return estimate errors against truth interpolated at estimate timestamps."""

    required_columns = {"time_s", "east_m", "north_m", "up_m"}
    if estimate_frame.empty or truth.empty:
        return pd.DataFrame()
    if not required_columns.issubset(estimate_frame.columns):
        return pd.DataFrame()
    if not required_columns.issubset(truth.columns):
        return pd.DataFrame()

    estimate_times = estimate_frame["time_s"].to_numpy(dtype=float)
    estimate_positions = estimate_frame[
        ["east_m", "north_m", "up_m"]
    ].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)

    finite_estimate = np.isfinite(estimate_times) & np.isfinite(
        estimate_positions
    ).all(axis=1)
    finite_truth = np.isfinite(truth_times) & np.isfinite(truth_positions).all(axis=1)
    if not finite_estimate.any() or not finite_truth.any():
        return pd.DataFrame()

    estimate_work = estimate_frame.loc[finite_estimate].copy()
    estimate_times = estimate_times[finite_estimate]
    estimate_positions = estimate_positions[finite_estimate]
    truth_times = truth_times[finite_truth]
    truth_positions = truth_positions[finite_truth]

    truth_at_estimate, valid = _metrics.interpolate_positions_at_times(
        truth_times,
        truth_positions,
        estimate_times,
        max_time_delta_s=max_eval_time_delta_s,
    )
    deltas = estimate_positions - truth_at_estimate
    error_2d = np.linalg.norm(deltas[:, :2], axis=1)
    error_3d = np.linalg.norm(deltas, axis=1)
    finite = valid & np.isfinite(error_2d) & np.isfinite(error_3d)

    out = estimate_work.loc[finite].copy()
    nearest_truth_indices = _metrics.nearest_time_indices(truth_times, estimate_times)
    out["truth_time_delta_s"] = np.abs(
        truth_times[nearest_truth_indices] - estimate_times
    )[finite]
    out["error_2d_m"] = error_2d[finite]
    out["error_3d_m"] = error_3d[finite]
    return out


def install() -> None:
    """Install interpolation-based alignment before sequence scoping wraps it."""

    if getattr(_diagnostics, "_truth_interpolation_patch_applied", False):
        return
    _diagnostics._position_error_frame = _position_error_frame
    implementation = getattr(_diagnostics, "_IMPL", None)
    if implementation is not None:
        implementation._position_error_frame = _position_error_frame
    _diagnostics._truth_interpolation_patch_applied = True


install()
