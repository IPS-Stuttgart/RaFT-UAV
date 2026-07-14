"""Compatibility package for sequence-safe radar oracle diagnostics.

The maintained implementation lives in the sibling
``radar_oracle_diagnostics.py`` module. This package preserves the public import
path while keeping pooled sequences and partially indexed radar frames distinct.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_oracle_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._radar_oracle_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load radar oracle diagnostics from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed nullable sequence identifiers for pooled diagnostics."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    missing = keys.isna() | keys.eq("") | keys.str.lower().isin({"nan", "none", "<na>"})
    return keys.mask(missing)


def _radar_frame_key_values(frame: pd.DataFrame) -> pd.Series:
    """Choose a complete frame-index key, otherwise rounded timestamps."""

    if "frame_index" in frame.columns:
        frame_indices = pd.to_numeric(frame["frame_index"], errors="coerce")
        if bool(np.isfinite(frame_indices.to_numpy(dtype=float)).all()):
            return frame_indices
    if "time_s" not in frame.columns:
        raise KeyError("radar is missing both frame_index and time_s")
    return pd.to_numeric(frame["time_s"], errors="coerce").round(9)


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Return physical radar frames without crossing sequence boundaries."""

    if radar.empty:
        return []
    work = radar.copy()
    group_columns: list[str] = []
    if "sequence_id" in work.columns:
        work["_sequence_key"] = _sequence_keys(work["sequence_id"])
        group_columns.append("_sequence_key")
    work["_frame_key"] = _radar_frame_key_values(work)
    group_columns.append("_frame_key")
    sort_columns = [
        *group_columns,
        *[
            column
            for column in ("time_s", "frame_index", "track_id", "track_index")
            if column in work.columns
        ],
    ]
    work = work.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    group_key: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else group_columns
    )
    return [
        rows.drop(columns=["_sequence_key", "_frame_key"], errors="ignore").copy()
        for _, rows in work.groupby(group_key, sort=True, dropna=False)
    ]


def _matching_truth_rows(truth: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict truth to the frame's normalized sequence when labels are available."""

    if "sequence_id" not in frame.columns or "sequence_id" not in truth.columns:
        return truth
    frame_keys = _sequence_keys(frame["sequence_id"])
    unique_keys = pd.unique(frame_keys)
    if len(unique_keys) != 1:
        return truth.iloc[0:0].copy()
    sequence_key = unique_keys[0]
    truth_keys = _sequence_keys(truth["sequence_id"])
    if pd.isna(sequence_key):
        mask = truth_keys.isna()
    else:
        mask = truth_keys.eq(sequence_key).fillna(False)
    return truth.loc[mask].copy()


def nearest_candidate_oracle(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    time_offset_s: float = 0.0,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Select the truth-nearest candidate independently for each physical frame."""

    if radar.empty:
        return _IMPL._empty_oracle_selection(radar)
    required = {"time_s", "east_m", "north_m", "up_m"}
    if not required.issubset(radar.columns):
        raise KeyError(
            f"radar is missing required columns: {sorted(required - set(radar.columns))}"
        )
    rows: list[pd.Series] = []
    for frame in _radar_frame_groups(radar):
        frame_time = float(pd.to_numeric(frame["time_s"], errors="coerce").median())
        frame_truth = _matching_truth_rows(truth, frame)
        if frame_truth.empty:
            continue
        truth_position, valid = _IMPL.interpolate_truth_positions(
            frame_truth,
            [frame_time + float(time_offset_s)],
            max_time_delta_s=max_time_delta_s,
        )
        if not bool(valid[0]):
            continue
        xyz = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
        finite = np.isfinite(xyz).all(axis=1)
        if not finite.any():
            continue
        errors_3d = np.full(len(frame), np.inf, dtype=float)
        errors_2d = np.full(len(frame), np.inf, dtype=float)
        residuals = xyz[finite] - truth_position[0]
        errors_3d[finite] = np.linalg.norm(residuals, axis=1)
        errors_2d[finite] = np.linalg.norm(residuals[:, :2], axis=1)
        best = int(np.argmin(errors_3d))
        selected = frame.iloc[best].copy()
        selected["oracle_time_offset_s"] = float(time_offset_s)
        selected["oracle_truth_time_s"] = frame_time + float(time_offset_s)
        selected["oracle_error_3d_m"] = float(errors_3d[best])
        selected["oracle_error_2d_m"] = float(errors_2d[best])
        selected["oracle_candidate_rows"] = int(len(frame))
        selected["association_mode"] = "oracle-nearest-candidate"
        rows.append(selected)
    if not rows:
        return _IMPL._empty_oracle_selection(radar)
    selected = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in (
            "sequence_id",
            "time_s",
            "frame_index",
            "track_id",
            "track_index",
        )
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


_IMPL._radar_frame_groups = _radar_frame_groups
_IMPL.nearest_candidate_oracle = nearest_candidate_oracle

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_keys"] = _sequence_keys
globals()["_radar_frame_key_values"] = _radar_frame_key_values
globals()["_radar_frame_groups"] = _radar_frame_groups
globals()["_matching_truth_rows"] = _matching_truth_rows
globals()["nearest_candidate_oracle"] = nearest_candidate_oracle

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
