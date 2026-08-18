"""Compatibility validation for retention-aware Fortem track identifiers.

The maintained implementation lives in the sibling
``tracklet_viterbi_retention.py`` module. This package preserves the public
import path while ensuring malformed track identifiers cannot manufacture
support for an unrelated integer track and duplicate rows cannot manufacture
extra historical support for one physical radar frame.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from statistics import median
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi_retention.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._tracklet_viterbi_retention_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load retention-aware tracklet-Viterbi implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACK_SUPPORT_BY_ID = _IMPL._track_support_by_id


def _finite_float(value: object) -> float | None:
    """Return a finite scalar float, or ``None`` for malformed metadata."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _physical_frame_key(row: pd.Series, *, row_position: int) -> tuple[object, ...]:
    """Return a collision-safe physical-frame identity for support counting."""

    frame_index = _finite_float(row.get("frame_index"))
    time_s = _finite_float(row.get("time_s"))
    if frame_index is not None and time_s is not None:
        return ("frame_index_time", frame_index, time_s)
    if frame_index is not None:
        return ("frame_index", frame_index)
    if time_s is not None:
        return ("time_s", time_s)
    return ("row", int(row_position))


def _collapse_duplicate_track_frames(radar: pd.DataFrame) -> pd.DataFrame:
    """Keep one support sample per exact track ID and physical radar frame.

    Repeated rows for the same track/frame are data duplicates rather than
    independent persistence evidence. Their UAV class probabilities are reduced
    to one per-frame median so duplicates cannot change either the support count
    or the class-probability contribution to the support score.
    """

    if radar.empty:
        return radar.copy()
    grouped_positions: dict[tuple[int, tuple[object, ...]], list[int]] = {}
    for position, (_, row) in enumerate(radar.iterrows()):
        key = (
            int(row["track_id"]),
            _physical_frame_key(row, row_position=position),
        )
        grouped_positions.setdefault(key, []).append(position)

    collapsed_rows: list[pd.Series] = []
    for positions in grouped_positions.values():
        group = radar.iloc[positions]
        representative = group.iloc[0].copy()
        if "cat_prob_uav" in group.columns:
            values = [
                number
                for value in group["cat_prob_uav"]
                if (number := _finite_float(value)) is not None
            ]
            representative["cat_prob_uav"] = float(median(values)) if values else float("nan")
        collapsed_rows.append(representative)
    return pd.DataFrame(collapsed_rows, columns=radar.columns).reset_index(drop=True)


def _track_support_by_id(radar: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Return support for exact integer IDs using distinct physical frames."""

    if radar.empty or "track_id" not in radar.columns:
        return _ORIGINAL_TRACK_SUPPORT_BY_ID(radar)
    track_ids = pd.Series(
        [_IMPL._base._optional_track_id(value) for value in radar["track_id"]],
        index=radar.index,
        dtype="Int64",
    )
    valid = track_ids.notna()
    if not bool(valid.any()):
        return {}
    normalized = radar.loc[valid].copy()
    normalized["track_id"] = track_ids.loc[valid].astype(int).to_numpy()
    normalized = _collapse_duplicate_track_frames(normalized)
    return _ORIGINAL_TRACK_SUPPORT_BY_ID(normalized)


_IMPL._track_support_by_id = _track_support_by_id

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_TRACK_SUPPORT_BY_ID"] = _ORIGINAL_TRACK_SUPPORT_BY_ID
globals()["_track_support_by_id"] = _track_support_by_id

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
