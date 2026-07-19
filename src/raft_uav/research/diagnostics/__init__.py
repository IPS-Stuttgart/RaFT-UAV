"""Compatibility fixes for research diagnostic frame and track identifiers.

The maintained implementation lives in the sibling ``diagnostics.py`` module.
This package preserves the public import path while retaining partially indexed
radar frames and preventing fractional identifiers from being truncated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.research._diagnostics_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load research diagnostics from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

_ORIGINAL_TRACK_SWITCH_METRICS = _LEGACY.track_switch_metrics


def _event_index_value(value: object) -> int | float | None:
    """Normalize a finite frame index without truncating fractional values."""

    integer = _optional_int(value)
    if integer is not None:
        return integer
    return _optional_float(value)


def _radar_frame_groups(
    radar: pd.DataFrame,
) -> list[tuple[tuple[str, int | float], pd.DataFrame]]:
    """Group rows by exact frame index, falling back to time per row."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns, kind="mergesort").copy()
    frame_values = (
        ordered["frame_index"].tolist()
        if "frame_index" in ordered.columns
        else [None] * len(ordered)
    )
    time_values = (
        ordered["time_s"].tolist()
        if "time_s" in ordered.columns
        else [None] * len(ordered)
    )

    group_keys: list[tuple[str, int | float] | None] = []
    for frame_index, time_s in zip(frame_values, time_values, strict=True):
        event_index = _event_index_value(frame_index)
        if event_index is not None:
            group_keys.append(("frame_index", event_index))
            continue
        event_time = _optional_float(time_s)
        group_keys.append(
            None if event_time is None else ("time_s", round(event_time, 9))
        )

    key_column = "_research_diagnostic_frame_key"
    ordered[key_column] = group_keys
    ordered = ordered.loc[ordered[key_column].notna()].copy()

    groups: list[tuple[tuple[str, int | float], pd.DataFrame]] = []
    for event_key, group in ordered.groupby(key_column, sort=False):
        groups.append(
            (
                event_key,
                group.drop(columns=key_column).copy(),
            )
        )
    return groups


def _radar_event_key(frame: pd.DataFrame) -> tuple[str, int | float]:
    """Return the exact frame key used by ``_radar_frame_groups``."""

    if "frame_index" in frame.columns and not frame.empty:
        for value in frame["frame_index"]:
            event_index = _event_index_value(value)
            if event_index is not None:
                return ("frame_index", event_index)
    return ("time_s", round(float(frame["time_s"].median()), 9))


def _row_event_key(row: pd.Series) -> tuple[str, int | float]:
    """Return an exact selected-row key without integer truncation."""

    if "frame_index" in row.index:
        event_index = _event_index_value(row.get("frame_index"))
        if event_index is not None:
            return ("frame_index", event_index)
    return ("time_s", round(float(row["time_s"]), 9))


def track_switch_metrics(
    selected: pd.DataFrame,
    *,
    long_gap_s: float = 5.0,
) -> dict[str, object]:
    """Compute switch metrics without inventing IDs from malformed values."""

    normalized = selected.copy()
    if "track_id" in normalized.columns:
        normalized["track_id"] = normalized["track_id"].map(_optional_int)
    return _ORIGINAL_TRACK_SWITCH_METRICS(normalized, long_gap_s=long_gap_s)


_LEGACY._optional_float = _optional_float
_LEGACY._optional_int = _optional_int
_LEGACY._radar_frame_groups = _radar_frame_groups
_LEGACY._radar_event_key = _radar_event_key
_LEGACY._row_event_key = _row_event_key
_LEGACY.track_switch_metrics = track_switch_metrics

for _name in dir(_LEGACY):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_LEGACY, _name)
globals()["_event_index_value"] = _event_index_value
globals()["_radar_frame_groups"] = _radar_frame_groups
globals()["_radar_event_key"] = _radar_event_key
globals()["_row_event_key"] = _row_event_key
globals()["track_switch_metrics"] = track_switch_metrics

__doc__ = _LEGACY.__doc__
__all__ = [
    name
    for name in dir(_LEGACY)
    if not (name.startswith("__") and name.endswith("__"))
]
