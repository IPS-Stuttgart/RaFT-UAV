"""Compatibility package for sequence-safe radar oracle diagnostics.

The maintained implementation lives in the sibling
``radar_oracle_diagnostics.py`` module. This package preserves the public import
path while keeping pooled sequences and partially indexed radar frames distinct.
Duplicate truth timestamps follow the shared final-sample trajectory convention.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int

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
_ORIGINAL_INTERPOLATE_TRUTH_POSITIONS = _IMPL.interpolate_truth_positions
_ORIGINAL_TIME_OFFSET_SWEEP = _IMPL.time_offset_sweep


def _finite_real_scalar(value: Any, *, field: str) -> float:
    """Return a finite real scalar without Boolean or array coercion."""

    message = f"{field} must be a finite real scalar"
    seen: set[int] = set()
    scalar = value
    while isinstance(scalar, np.ndarray):
        if np.ma.is_masked(scalar) or scalar.ndim != 0:
            raise ValueError(message)
        marker = id(scalar)
        if marker in seen:
            raise ValueError(message)
        seen.add(marker)
        scalar = scalar.item()
    if np.ma.is_masked(scalar) or isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(scalar, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _time_offset_seconds(value: Any) -> float:
    """Validate one signed timestamp correction."""

    return _finite_real_scalar(value, field="time_offset_s")


def _max_time_delta_seconds(value: Any | None) -> float | None:
    """Validate an optional non-negative truth freshness gate."""

    if value is None:
        return None
    maximum = _finite_real_scalar(value, field="max_time_delta_s")
    if maximum < 0.0:
        raise ValueError("max_time_delta_s must be nonnegative or None")
    return maximum


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed nullable sequence identifiers for pooled diagnostics."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    missing = keys.isna() | keys.eq("") | keys.str.lower().isin({"nan", "none", "<na>"})
    return keys.mask(missing)


def _sequence_metadata_state(
    frame: pd.DataFrame,
    *,
    name: str,
) -> tuple[bool, set[str]]:
    """Return whether sequence metadata are complete and their explicit identifiers."""

    if "sequence_id" not in frame.columns or frame.empty:
        return False, set()
    keys = _sequence_keys(frame["sequence_id"])
    explicit = keys.dropna()
    if explicit.empty:
        return False, set()
    missing_mask = keys.isna().to_numpy(dtype=bool)
    if bool(missing_mask.any()):
        missing_positions = np.flatnonzero(missing_mask).tolist()
        raise ValueError(
            f"{name} sequence_id is partially populated; missing row positions: "
            f"{missing_positions}"
        )
    return True, set(explicit.astype(str).tolist())


def _validate_sequence_metadata(radar: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Reject ambiguous pooled or partially labeled sequence metadata."""

    radar_labeled, radar_sequences = _sequence_metadata_state(radar, name="radar")
    truth_labeled, truth_sequences = _sequence_metadata_state(truth, name="truth")
    if radar_labeled and not truth_labeled and len(radar_sequences) > 1:
        raise ValueError(
            "pooled radar sequence_id metadata require truth sequence_id metadata"
        )
    if truth_labeled and not radar_labeled and len(truth_sequences) > 1:
        raise ValueError(
            "pooled truth sequence_id metadata require radar sequence_id metadata"
        )


def _radar_frame_key_values(frame: pd.DataFrame) -> pd.Series:
    """Use frame index and time together, falling back to either usable field."""

    if "frame_index" not in frame.columns and "time_s" not in frame.columns:
        raise KeyError("radar is missing both frame_index and time_s")
    frame_indices = (
        frame["frame_index"].tolist()
        if "frame_index" in frame.columns
        else [None] * len(frame)
    )
    time_values = (
        frame["time_s"].tolist()
        if "time_s" in frame.columns
        else [None] * len(frame)
    )
    keys: list[tuple[object, ...] | None] = []
    for frame_index, time_s in zip(frame_indices, time_values, strict=True):
        event_index = _optional_int(frame_index)
        event_time = _optional_float(time_s)
        if event_index is not None and event_time is not None:
            keys.append(("frame_index_time", event_index, round(event_time, 9)))
        elif event_index is not None:
            keys.append(("frame_index", event_index))
        elif event_time is not None:
            keys.append(("time_s", round(event_time, 9)))
        else:
            keys.append(None)
    return pd.Series(keys, index=frame.index, dtype=object)


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
    work = work.loc[work["_frame_key"].notna()].copy()
    if work.empty:
        return []
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
    truth_keys = _sequence_keys(truth["sequence_id"])
    if frame_keys.dropna().empty or truth_keys.dropna().empty:
        return truth
    unique_keys = pd.unique(frame_keys)
    if len(unique_keys) != 1:
        return truth.iloc[0:0].copy()
    sequence_key = unique_keys[0]
    if pd.isna(sequence_key):
        mask = truth_keys.isna()
    else:
        mask = truth_keys.eq(sequence_key).fillna(False)
    return truth.loc[mask].copy()


def _truth_with_final_duplicate_samples(truth: pd.DataFrame) -> pd.DataFrame:
    """Keep the final row at each finite truth timestamp.

    Truth tables can contain sequential posterior snapshots at one timestamp.
    The final row is the authoritative trajectory sample throughout the shared
    metric layer; retaining an earlier duplicate here would make oracle
    interpolation and nearest-sample metrics disagree.
    """

    if truth.empty or "time_s" not in truth.columns:
        return truth
    time_keys = pd.Series(
        [_optional_float(value) for value in truth["time_s"]],
        index=truth.index,
        dtype=float,
    )
    finite = time_keys.notna()
    duplicate = pd.Series(False, index=truth.index, dtype=bool)
    duplicate.loc[finite] = time_keys.loc[finite].duplicated(keep="last")
    return truth.loc[~duplicate].copy()


def interpolate_truth_positions(
    truth: pd.DataFrame,
    query_times_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate truth after applying the shared final-duplicate convention."""

    return _ORIGINAL_INTERPOLATE_TRUTH_POSITIONS(
        _truth_with_final_duplicate_samples(truth),
        query_times_s,
        max_time_delta_s=_max_time_delta_seconds(max_time_delta_s),
    )


def nearest_candidate_oracle(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    time_offset_s: float = 0.0,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Select the truth-nearest candidate independently for each physical frame."""

    offset = _time_offset_seconds(time_offset_s)
    maximum_delta = _max_time_delta_seconds(max_time_delta_s)
    if radar.empty:
        return _IMPL._empty_oracle_selection(radar)
    required = {"time_s", "east_m", "north_m", "up_m"}
    if not required.issubset(radar.columns):
        raise KeyError(
            f"radar is missing required columns: {sorted(required - set(radar.columns))}"
        )
    _validate_sequence_metadata(radar, truth)
    rows: list[pd.Series] = []
    for frame in _radar_frame_groups(radar):
        frame_time = float(pd.to_numeric(frame["time_s"], errors="coerce").median())
        frame_truth = _matching_truth_rows(truth, frame)
        if frame_truth.empty:
            continue
        truth_position, valid = interpolate_truth_positions(
            frame_truth,
            [frame_time + offset],
            max_time_delta_s=maximum_delta,
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
        selected["oracle_time_offset_s"] = offset
        selected["oracle_truth_time_s"] = frame_time + offset
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


def time_offset_sweep(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Sweep only validated timestamp corrections and freshness settings."""

    offsets = [_time_offset_seconds(offset) for offset in offsets_s]
    return _ORIGINAL_TIME_OFFSET_SWEEP(
        radar,
        truth,
        offsets,
        max_time_delta_s=_max_time_delta_seconds(max_time_delta_s),
    )


_IMPL.interpolate_truth_positions = interpolate_truth_positions
_IMPL._radar_frame_groups = _radar_frame_groups
_IMPL.nearest_candidate_oracle = nearest_candidate_oracle
_IMPL.time_offset_sweep = time_offset_sweep

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_real_scalar"] = _finite_real_scalar
globals()["_time_offset_seconds"] = _time_offset_seconds
globals()["_max_time_delta_seconds"] = _max_time_delta_seconds
globals()["_sequence_keys"] = _sequence_keys
globals()["_sequence_metadata_state"] = _sequence_metadata_state
globals()["_validate_sequence_metadata"] = _validate_sequence_metadata
globals()["_radar_frame_key_values"] = _radar_frame_key_values
globals()["_radar_frame_groups"] = _radar_frame_groups
globals()["_matching_truth_rows"] = _matching_truth_rows
globals()["_truth_with_final_duplicate_samples"] = _truth_with_final_duplicate_samples
globals()["interpolate_truth_positions"] = interpolate_truth_positions
globals()["nearest_candidate_oracle"] = nearest_candidate_oracle
globals()["time_offset_sweep"] = time_offset_sweep

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
