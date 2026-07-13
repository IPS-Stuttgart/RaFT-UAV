"""Compatibility wrapper hardening the Track 5 jerk-limit implementation.

The maintained implementation lives in the sibling ``track5_jerk_limit.py``
module. This package keeps the public import path while preserving the fixed
trajectory grid and ensuring that jerk values remain attached to the actual
four rows of every valid finite-difference window when earlier windows are
skipped because of duplicate timestamps.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_jerk_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_jerk_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 jerk-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _normalized_submission_rejecting_invalid_rows(
    submission: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a submission without silently deleting malformed grid rows."""

    rows = pd.DataFrame(submission).copy()
    lower = {str(column).strip().lower() for column in rows.columns}
    if {"sequence", "timestamp", "position", "classification"}.issubset(lower):
        official = _IMPL.normalize_official_track5_results_frame(rows)
        positions = [
            _IMPL.parse_official_position_cell(value)
            for value in official["Position"]
        ]
        xyz = pd.DataFrame(
            positions,
            columns=["state_x_m", "state_y_m", "state_z_m"],
            index=official.index,
        )
        rows = pd.DataFrame(
            {
                "sequence_id": official["Sequence"].astype(str),
                "time_s": pd.to_numeric(official["Timestamp"], errors="coerce"),
                "state_x_m": xyz["state_x_m"],
                "state_y_m": xyz["state_y_m"],
                "state_z_m": xyz["state_z_m"],
                "Classification": official["Classification"],
            }
        )
    required = {
        "sequence_id",
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "Classification",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"submission missing normalized columns: {sorted(missing)}")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    numeric_columns = (
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "Classification",
    )
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite_columns = ("time_s", "state_x_m", "state_y_m", "state_z_m")
    finite = np.isfinite(rows[list(finite_columns)].to_numpy(float)).all(axis=1)
    if not finite.all():
        invalid_indices = rows.index[~finite].tolist()
        preview = ", ".join(str(index) for index in invalid_indices[:5])
        suffix = ", ..." if len(invalid_indices) > 5 else ""
        raise ValueError(
            "submission contains non-finite time or position values "
            f"at row indices: {preview}{suffix}"
        )
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _row_jerk_proxy_with_window_support(
    times: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """Map every valid jerk window to the rows in that window."""

    count = len(times)
    row_jerk = np.full(count, np.nan, dtype=float)
    d3 = _IMPL._third_derivative_matrix(times)
    if d3.size == 0:
        return row_jerk
    jerk_windows = d3 @ np.asarray(xyz, dtype=float)
    norms = np.linalg.norm(jerk_windows, axis=1)
    for coefficients, norm in zip(d3, norms, strict=True):
        for row_index in np.flatnonzero(coefficients):
            if np.isnan(row_jerk[row_index]) or norm > row_jerk[row_index]:
                row_jerk[row_index] = float(norm)
    return row_jerk


_IMPL._normalized_submission = _normalized_submission_rejecting_invalid_rows
_IMPL._row_jerk_proxy = _row_jerk_proxy_with_window_support

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
