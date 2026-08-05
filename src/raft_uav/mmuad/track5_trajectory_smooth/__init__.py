"""Compatibility wrapper for Track 5 trajectory-smoother validation.

The maintained implementation lives in the sibling ``track5_trajectory_smooth.py``
module. This package keeps the public import path while rejecting malformed
fixed-grid rows, duplicate ``(sequence_id, time_s)`` keys, and lossy programmatic
smoothing controls before coercion can change their meaning.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.submission import (
    parse_official_classification_cell,
    parse_official_sequence_cell,
)
from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_trajectory_smooth.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_trajectory_smooth_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 trajectory smoother from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_NORMALIZE = _IMPL._normalized_estimate_rows
_ORIGINAL_SMOOTH = _IMPL.smooth_track5_submission_rows

_REQUIRED_ROW_COLUMNS = {
    "sequence_id",
    "time_s",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "Classification",
}
_NUMERIC_ROW_COLUMNS = ("time_s", "state_x_m", "state_y_m", "state_z_m")


def _normalized_estimate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows without silently deleting or relabeling malformed cells."""

    validated = pd.DataFrame(rows).copy()
    missing = sorted(_REQUIRED_ROW_COLUMNS.difference(validated.columns))
    if missing:
        raise ValueError(f"Track 5 rows missing normalized columns: {missing}")

    sequence_ids: list[str] = []
    invalid_sequence_rows: list[Any] = []
    for index, value in validated["sequence_id"].items():
        try:
            sequence_ids.append(parse_official_sequence_cell(value))
        except ValueError:
            sequence_ids.append("")
            invalid_sequence_rows.append(index)
    if invalid_sequence_rows:
        raise ValueError(
            "Track 5 rows contain invalid sequence_id values at row indices: "
            f"{_row_index_preview(invalid_sequence_rows)}"
        )
    validated["sequence_id"] = sequence_ids

    classifications: list[int] = []
    invalid_classification_rows: list[Any] = []
    for index, value in validated["Classification"].items():
        try:
            classifications.append(parse_official_classification_cell(value))
        except ValueError:
            classifications.append(0)
            invalid_classification_rows.append(index)
    if invalid_classification_rows:
        raise ValueError(
            "Track 5 rows contain invalid Classification values at row indices: "
            f"{_row_index_preview(invalid_classification_rows)}"
        )
    validated["Classification"] = classifications

    for column in _NUMERIC_ROW_COLUMNS:
        normalized_values: list[float] = []
        invalid_rows: list[Any] = []
        for index, value in validated[column].items():
            normalized = optional_float(value)
            if normalized is None:
                normalized_values.append(float("nan"))
                invalid_rows.append(index)
            else:
                normalized_values.append(normalized)
        if invalid_rows:
            raise ValueError(
                f"Track 5 rows contain invalid {column} values at row indices: "
                f"{_row_index_preview(invalid_rows)}"
            )
        validated[column] = normalized_values

    normalized = _ORIGINAL_NORMALIZE(validated)
    if len(normalized) != len(validated):
        raise RuntimeError("Track 5 row normalization unexpectedly changed row count")

    duplicate = normalized.duplicated(["sequence_id", "time_s"], keep=False)
    if duplicate.any():
        keys = (
            normalized.loc[duplicate, ["sequence_id", "time_s"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        preview = ", ".join(f"({sequence_id!r}, {time_s!r})" for sequence_id, time_s in keys)
        raise ValueError(f"Track 5 rows contain duplicate sequence/timestamp keys: {preview}")
    return normalized


def smooth_track5_submission_rows(
    rows: pd.DataFrame,
    *,
    window_s: float = 15.0,
    bandwidth_s: float | None = None,
    blend: float = 1.0,
    max_correction_m: float | None = 10.0,
    min_neighbors: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate scalar controls losslessly before delegating to the smoother."""

    normalized_window_s = _positive_finite_float(window_s, name="window_s")
    normalized_bandwidth_s = (
        None
        if bandwidth_s is None
        else _positive_finite_float(bandwidth_s, name="bandwidth_s")
    )
    normalized_blend = optional_float(blend)
    if normalized_blend is None or not 0.0 <= normalized_blend <= 1.0:
        raise ValueError("blend must be a finite real scalar in [0, 1]")
    normalized_max_correction_m = optional_float(max_correction_m)
    if max_correction_m is not None and (
        normalized_max_correction_m is None or normalized_max_correction_m < 0.0
    ):
        raise ValueError("max_correction_m must be a finite non-negative real scalar")
    normalized_min_neighbors = optional_int(min_neighbors)
    if normalized_min_neighbors is None or normalized_min_neighbors < 1:
        raise ValueError("min_neighbors must be a positive integer")

    return _ORIGINAL_SMOOTH(
        rows,
        window_s=normalized_window_s,
        bandwidth_s=normalized_bandwidth_s,
        blend=normalized_blend,
        max_correction_m=normalized_max_correction_m,
        min_neighbors=normalized_min_neighbors,
    )


def _positive_finite_float(value: object, *, name: str) -> float:
    normalized = optional_float(value)
    if normalized is None or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return normalized


def _row_index_preview(indices: list[Any]) -> str:
    preview = ", ".join(str(index) for index in indices[:5])
    return f"{preview}, ..." if len(indices) > 5 else preview


_IMPL._normalized_estimate_rows = _normalized_estimate_rows
_IMPL.smooth_track5_submission_rows = smooth_track5_submission_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_estimate_rows"] = _normalized_estimate_rows
globals()["smooth_track5_submission_rows"] = smooth_track5_submission_rows
globals()["_row_index_preview"] = _row_index_preview

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
