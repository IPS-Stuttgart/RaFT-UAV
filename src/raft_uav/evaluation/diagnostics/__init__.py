"""Compatibility wrapper that scopes pooled diagnostics by sequence."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load diagnostic summary implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACK_SWITCH_SUMMARY = _IMPL._track_switch_summary
_ORIGINAL_POSITION_ERROR_FRAME = _IMPL._position_error_frame


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed nullable sequence IDs without stringifying missing values."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    return keys.where(keys.notna() & keys.ne(""))


def _sequence_mask(keys: pd.Series, sequence: object) -> pd.Series:
    if pd.isna(sequence):
        return keys.isna()
    return keys.eq(sequence).fillna(False)


def _temporary_position_column(columns: pd.Index) -> str:
    name = "__raft_uav_diagnostic_input_position__"
    while name in columns:
        name += "_"
    return name


def _track_switch_summary(frame: pd.DataFrame, *, top_n: int) -> dict[str, Any]:
    """Count switches within sequences, not across pooled sequence boundaries."""

    if frame.empty or "track_id" not in frame.columns or "sequence_id" not in frame.columns:
        return _ORIGINAL_TRACK_SWITCH_SUMMARY(frame, top_n=top_n)

    work = pd.DataFrame(frame).copy()
    sequence_keys = _sequence_keys(work["sequence_id"])
    transitions: Counter[tuple[int, int]] = Counter()
    events: list[dict[str, Any]] = []
    updates_with_track_id = 0
    first_track_id: int | None = None
    last_track_id: int | None = None

    for sequence in pd.unique(sequence_keys):
        group = work.loc[_sequence_mask(sequence_keys, sequence)]
        summary = _ORIGINAL_TRACK_SWITCH_SUMMARY(
            group,
            top_n=max(top_n, len(group)),
        )
        updates_with_track_id += int(summary["updates_with_track_id"])
        if first_track_id is None and summary["first_track_id"] is not None:
            first_track_id = int(summary["first_track_id"])
        if summary["last_track_id"] is not None:
            last_track_id = int(summary["last_track_id"])
        for transition in summary["top_transitions"]:
            key = (
                int(transition["from_track_id"]),
                int(transition["to_track_id"]),
            )
            transitions[key] += int(transition["count"])
        for event in summary["events"]:
            payload = dict(event)
            if not pd.isna(sequence):
                payload["sequence_id"] = str(sequence)
            events.append(payload)

    numeric_ids = pd.to_numeric(work["track_id"], errors="coerce")
    finite_ids = numeric_ids[np.isfinite(numeric_ids)].astype(int)
    return {
        "count": int(sum(transitions.values())),
        "updates_with_track_id": int(updates_with_track_id),
        "unique_track_ids": int(finite_ids.nunique()),
        "first_track_id": first_track_id,
        "last_track_id": last_track_id,
        "top_transitions": [
            {
                "from_track_id": int(source),
                "to_track_id": int(destination),
                "count": int(count),
            }
            for (source, destination), count in transitions.most_common(top_n)
        ],
        "events": events[:top_n],
    }


def _position_error_frame(
    *,
    estimate_frame: pd.DataFrame,
    truth: pd.DataFrame,
    max_eval_time_delta_s: float | None,
) -> pd.DataFrame:
    """Match pooled estimates only to truth rows from the same sequence."""

    if "sequence_id" not in estimate_frame.columns or "sequence_id" not in truth.columns:
        return _ORIGINAL_POSITION_ERROR_FRAME(
            estimate_frame=estimate_frame,
            truth=truth,
            max_eval_time_delta_s=max_eval_time_delta_s,
        )

    prepared = pd.DataFrame(estimate_frame).copy()
    truth_rows = pd.DataFrame(truth).copy()
    position_column = _temporary_position_column(prepared.columns)
    prepared[position_column] = np.arange(len(prepared), dtype=int)
    estimate_keys = _sequence_keys(prepared["sequence_id"])
    truth_keys = _sequence_keys(truth_rows["sequence_id"])

    parts: list[pd.DataFrame] = []
    for sequence in pd.unique(estimate_keys):
        estimate_mask = _sequence_mask(estimate_keys, sequence)
        truth_mask = _sequence_mask(truth_keys, sequence)
        if not bool(truth_mask.any()):
            continue
        aligned = _ORIGINAL_POSITION_ERROR_FRAME(
            estimate_frame=prepared.loc[estimate_mask],
            truth=truth_rows.loc[truth_mask],
            max_eval_time_delta_s=max_eval_time_delta_s,
        )
        if not aligned.empty:
            parts.append(aligned)

    if not parts:
        return prepared.iloc[0:0].drop(columns=[position_column]).copy()

    result = pd.concat(parts, axis=0, sort=False)
    result = result.sort_values(position_column, kind="mergesort")
    return result.drop(columns=[position_column])


_IMPL._track_switch_summary = _track_switch_summary
_IMPL._position_error_frame = _position_error_frame

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_keys"] = _sequence_keys
globals()["_sequence_mask"] = _sequence_mask
globals()["_temporary_position_column"] = _temporary_position_column
globals()["_track_switch_summary"] = _track_switch_summary
globals()["_position_error_frame"] = _position_error_frame

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
