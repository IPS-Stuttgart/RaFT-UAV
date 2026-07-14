"""Compatibility package that scopes tracker truth errors by sequence.

The maintained implementation lives in the sibling ``tracker.py`` module. This
package preserves the public import path while preventing pooled trajectories
with overlapping timestamps from borrowing another sequence's truth rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracker.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._tracker_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load tracker implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_TRUTH_ERRORS = _IMPL.add_truth_errors
_TRUTH_ERROR_COLUMNS = (
    "truth_x_m",
    "truth_y_m",
    "truth_z_m",
    "error_2d_m",
    "error_3d_m",
)


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed nullable sequence identifiers for truth alignment."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    return keys.where(keys.notna() & keys.ne(""))


def add_truth_errors(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Attach truth errors without matching across sequence boundaries.

    Sequence-less inputs retain the historical global time-interpolation behavior.
    When both frames provide ``sequence_id``, each estimate is interpolated only
    against truth from the same normalized sequence. Rows without matching truth
    remain present with missing truth/error diagnostics.
    """

    estimate_rows = pd.DataFrame(estimates).copy()
    truth_rows = pd.DataFrame(truth).copy()
    if "sequence_id" not in estimate_rows.columns or "sequence_id" not in truth_rows.columns:
        return _ORIGINAL_ADD_TRUTH_ERRORS(estimate_rows, truth_rows)

    output = estimate_rows.drop(columns=list(_TRUTH_ERROR_COLUMNS), errors="ignore").copy()
    for column in _TRUTH_ERROR_COLUMNS:
        output[column] = np.nan

    estimate_keys = _sequence_keys(estimate_rows["sequence_id"])
    truth_keys = _sequence_keys(truth_rows["sequence_id"])
    for sequence_key in pd.unique(estimate_keys.dropna()):
        estimate_mask = estimate_keys.eq(sequence_key).fillna(False).to_numpy(dtype=bool)
        truth_mask = truth_keys.eq(sequence_key).fillna(False).to_numpy(dtype=bool)
        if not bool(estimate_mask.any()) or not bool(truth_mask.any()):
            continue
        estimate_positions = np.flatnonzero(estimate_mask)
        scored = _ORIGINAL_ADD_TRUTH_ERRORS(
            estimate_rows.iloc[estimate_positions].copy(),
            truth_rows.iloc[np.flatnonzero(truth_mask)].copy(),
        )
        for column in _TRUTH_ERROR_COLUMNS:
            output.iloc[estimate_positions, output.columns.get_loc(column)] = pd.to_numeric(
                scored[column],
                errors="coerce",
            ).to_numpy(float)
    return output


_IMPL.add_truth_errors = add_truth_errors

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_keys"] = _sequence_keys
globals()["add_truth_errors"] = add_truth_errors

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
