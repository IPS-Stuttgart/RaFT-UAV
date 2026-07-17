"""Compatibility fixes for research diagnostic frame and track identifiers.

The maintained implementation lives in the sibling ``diagnostics.py`` module.
This package preserves the public import path while retaining partially indexed
radar frames and preventing fractional identifiers from being truncated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
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
    """Group physical radar frames without dropping or merging valid indices."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns, kind="mergesort").copy()
    frame_indices = None
    if "frame_index" in ordered.columns:
        frame_indices = pd.Series(
            [_event_index_value(value) for value in ordered["frame_index"]],
            index=ordered.index,
            dtype=object,
        )
    use_frame_index = frame_indices is not None and bool(frame_indices.notna().all())
    if use_frame_index:
        ordered["_research_diagnostic_frame_key"] = frame_indices
        key_kind = "frame_index"
    else:
        times = pd.to_numeric(ordered["time_s"], errors="coerce")
        finite = np.isfinite(times.to_numpy(dtype=float))
        ordered = ordered.loc[finite].copy()
        ordered["_research_diagnostic_frame_key"] = times.loc[finite].to_numpy(
            dtype=float
        )
        key_kind = "time_s"

    groups: list[tuple[tuple[str, int | float], pd.DataFrame]] = []
    for key, group in ordered.groupby(
        "_research_diagnostic_frame_key",
        sort=True,
    ):
        if key_kind == "frame_index":
            event_index = _event_index_value(key)
            if event_index is None:  # pragma: no cover - guarded above
                continue
            event_key = ("frame_index", event_index)
        else:
            event_key = ("time_s", round(float(group["time_s"].median()), 9))
        groups.append(
            (
                event_key,
                group.drop(columns="_research_diagnostic_frame_key").copy(),
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
