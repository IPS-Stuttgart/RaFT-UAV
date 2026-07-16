"""Compatibility package for the basic MMUAD tracker."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tracker.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._tracker_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load tracker implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)

_ORIGINAL_CANDIDATE_ROWS_WITH_OPTIONAL_DEFAULTS = (
    _LEGACY._candidate_rows_with_optional_defaults
)
_TRACKER_NUMERIC_COLUMNS = (
    "time_s",
    "x_m",
    "y_m",
    "z_m",
    "std_xy_m",
    "std_z_m",
    "confidence",
)


def _candidate_rows_with_optional_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill optional columns and normalize values used numerically by the tracker."""

    out = _ORIGINAL_CANDIDATE_ROWS_WITH_OPTIONAL_DEFAULTS(rows)
    for column in _TRACKER_NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def add_truth_errors(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Attach interpolated truth errors only inside the finite truth time span."""

    out = estimates.copy()
    truth_interp = _LEGACY._finite_truth_by_time(truth)
    estimate_times = pd.to_numeric(out["time_s"], errors="coerce").to_numpy(float)
    interp = np.full((len(out), 3), np.nan, dtype=float)
    if not truth_interp.empty:
        truth_times = truth_interp["time_s"].to_numpy(float)
        truth_xyz = truth_interp[["x_m", "y_m", "z_m"]].to_numpy(float)
        supported = (
            np.isfinite(estimate_times)
            & (estimate_times >= truth_times[0])
            & (estimate_times <= truth_times[-1])
        )
        if supported.any():
            interp[supported] = np.column_stack(
                [
                    np.interp(
                        estimate_times[supported],
                        truth_times,
                        truth_xyz[:, axis],
                    )
                    for axis in range(3)
                ]
            )
    est_xyz = (
        out[["state_x_m", "state_y_m", "state_z_m"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    err = est_xyz - interp
    out["truth_x_m"] = interp[:, 0]
    out["truth_y_m"] = interp[:, 1]
    out["truth_z_m"] = interp[:, 2]
    out["error_2d_m"] = np.linalg.norm(err[:, :2], axis=1)
    out["error_3d_m"] = np.linalg.norm(err, axis=1)
    return out


_LEGACY._candidate_rows_with_optional_defaults = _candidate_rows_with_optional_defaults
_LEGACY.add_truth_errors = add_truth_errors
