"""Compatibility wrapper preserving the Track 5 temporal-repair grid.

The maintained implementation lives in the sibling ``track5_temporal_repair.py``
module. This package keeps the public import path while rejecting malformed
non-finite timestamps or positions instead of silently deleting those rows from
the fixed submission grid.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_temporal_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_temporal_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 temporal-repair implementation from {_IMPL_PATH}")
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


_IMPL._normalized_submission = _normalized_submission_rejecting_invalid_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
