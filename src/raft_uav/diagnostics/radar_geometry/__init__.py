"""Compatibility wrapper keeping radar-geometry summaries sequence-local.

The maintained implementation lives in the sibling ``radar_geometry.py``
module. This package preserves the public import path while preventing reused
track and frame identifiers from collapsing pooled flight diagnostics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_geometry.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._radar_geometry_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load radar geometry implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

for _name in dir(_IMPL):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_IMPL, _name)


def _normalized_sequence_id(value: object) -> str | None:
    """Return one normalized opaque sequence identifier or ``None``."""

    if value is None or np.ma.is_masked(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    normalized = str(value).strip()
    return normalized or None


def _sequence_scoped_integer_count(frame: pd.DataFrame, column: str) -> int:
    """Count integer identifiers independently within each explicit sequence."""

    if column not in frame.columns:
        return 0
    if "sequence_id" not in frame.columns:
        return _IMPL._unique_integer_count(frame[column])

    values: set[tuple[str | None, int]] = set()
    for sequence_id, identifier in zip(
        frame["sequence_id"].tolist(),
        frame[column].tolist(),
        strict=True,
    ):
        parsed = optional_int(identifier)
        if parsed is not None:
            values.add((_normalized_sequence_id(sequence_id), parsed))
    return len(values)


def summarize_radar_geometry_audit(audit: pd.DataFrame) -> dict[str, object]:
    """Summarize pooled audit rows without merging reused flight-local IDs."""

    summary = _IMPL.summarize_radar_geometry_audit(audit)
    if "sequence_id" in audit.columns:
        if "track_id" in audit.columns:
            summary["track_ids"] = _sequence_scoped_integer_count(audit, "track_id")
        if "frame_index" in audit.columns:
            summary["frames"] = _sequence_scoped_integer_count(audit, "frame_index")
    return summary


def summarize_radar_geometry_by_track(audit: pd.DataFrame) -> pd.DataFrame:
    """Return one summary row per sequence-local Fortem track ID."""

    if audit.empty or "track_id" not in audit.columns:
        return pd.DataFrame()
    if "sequence_id" not in audit.columns:
        return _IMPL.summarize_radar_geometry_by_track(audit)

    sequence_column = "__raft_uav_sequence_id__"
    while sequence_column in audit.columns:
        sequence_column = f"_{sequence_column}"

    working = audit.copy()
    working[sequence_column] = [
        _normalized_sequence_id(value) for value in working["sequence_id"].tolist()
    ]
    summaries: list[pd.DataFrame] = []
    for sequence_id, sequence_rows in working.groupby(
        sequence_column,
        sort=False,
        dropna=False,
    ):
        per_track = _IMPL.summarize_radar_geometry_by_track(
            sequence_rows.drop(columns=[sequence_column])
        )
        if per_track.empty:
            continue
        per_track.insert(0, "sequence_id", sequence_id)
        summaries.append(per_track)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


_IMPL.summarize_radar_geometry_audit = summarize_radar_geometry_audit
_IMPL.summarize_radar_geometry_by_track = summarize_radar_geometry_by_track

globals()["_normalized_sequence_id"] = _normalized_sequence_id
globals()["_sequence_scoped_integer_count"] = _sequence_scoped_integer_count
globals()["summarize_radar_geometry_audit"] = summarize_radar_geometry_audit
globals()["summarize_radar_geometry_by_track"] = summarize_radar_geometry_by_track

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
