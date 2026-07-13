"""Compatibility wrapper that scopes uncertainty residual alignment by sequence.

The maintained implementation lives in the sibling ``uncertainty.py`` module.
This package preserves the public import surface while preventing pooled flights
with overlapping timestamps from borrowing another sequence's truth rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "uncertainty.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._uncertainty_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load uncertainty implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ALIGNED_RESIDUALS = _IMPL._aligned_residuals


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed, nullable sequence identifiers for alignment."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    return keys.where(keys.notna() & keys.ne(""))


def _temporary_position_column(columns: pd.Index) -> str:
    name = "__raft_uav_uncertainty_input_position__"
    while name in columns:
        name += "_"
    return name


def _aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Align measurements to nearest truth rows within each available sequence."""

    if "sequence_id" not in frame.columns or "sequence_id" not in truth.columns:
        return _ORIGINAL_ALIGNED_RESIDUALS(
            frame,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    prepared = pd.DataFrame(frame).copy()
    truth_rows = pd.DataFrame(truth).copy()
    position_column = _temporary_position_column(prepared.columns)
    prepared[position_column] = np.arange(len(prepared), dtype=int)
    frame_keys = _sequence_keys(prepared["sequence_id"])
    truth_keys = _sequence_keys(truth_rows["sequence_id"])

    parts: list[pd.DataFrame] = []
    for sequence_key in pd.unique(frame_keys.dropna()):
        frame_mask = frame_keys.eq(sequence_key).fillna(False)
        truth_mask = truth_keys.eq(sequence_key).fillna(False)
        if not bool(truth_mask.any()):
            continue
        aligned = _ORIGINAL_ALIGNED_RESIDUALS(
            prepared.loc[frame_mask],
            truth_rows.loc[truth_mask],
            max_time_delta_s=max_time_delta_s,
        )
        if not aligned.empty:
            parts.append(aligned)

    if not parts:
        return prepared.iloc[0:0].drop(columns=[position_column]).copy()

    result = pd.concat(parts, ignore_index=True, sort=False)
    result = result.sort_values(position_column, kind="mergesort")
    return result.drop(columns=[position_column]).reset_index(drop=True)


_IMPL._aligned_residuals = _aligned_residuals

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_keys"] = _sequence_keys
globals()["_temporary_position_column"] = _temporary_position_column
globals()["_aligned_residuals"] = _aligned_residuals

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
