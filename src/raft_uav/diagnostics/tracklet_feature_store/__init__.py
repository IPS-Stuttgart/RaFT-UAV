"""Compatibility fix for cross-run selected-radar candidate matching.

The maintained implementation lives in the sibling ``tracklet_feature_store.py``
module. This package preserves the public import path while matching external
selected-radar rows by stable track ID, with track-index fallback only when an
ID is unavailable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_feature_store.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._tracklet_feature_store_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load tracklet feature-store implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _identifier_key(
    row: pd.Series,
    identifier: str,
    value: int,
) -> tuple[object, ...]:
    """Return one frame-scoped candidate identifier key."""

    return (
        row.get("frame_key_type"),
        row.get("frame_key"),
        identifier,
        int(value),
    )


def _candidate_match_key(row: pd.Series) -> tuple[object, ...] | None:
    """Return the strongest available frame-scoped candidate identity."""

    track_id = _IMPL._optional_int(row.get("track_id"))
    if track_id is not None:
        return _identifier_key(row, "track_id", track_id)
    track_index = _IMPL._optional_int(row.get("track_index"))
    if track_index is not None:
        return _identifier_key(row, "track_index", track_index)
    return None


def _selection_mask(
    features: pd.DataFrame,
    selected_radar: pd.DataFrame | None,
) -> np.ndarray:
    """Match selected candidates without requiring unstable track-index parity."""

    if selected_radar is None or selected_radar.empty:
        return np.zeros(len(features), dtype=bool)

    selected = _IMPL._append_frame_keys(selected_radar)
    selected_track_ids: set[tuple[object, ...]] = set()
    selected_track_indices: set[tuple[object, ...]] = set()
    fallback_track_indices: set[tuple[object, ...]] = set()
    for _, row in selected.iterrows():
        track_id = _IMPL._optional_int(row.get("track_id"))
        track_index = _IMPL._optional_int(row.get("track_index"))
        if track_id is not None:
            selected_track_ids.add(_identifier_key(row, "track_id", track_id))
        if track_index is not None:
            index_key = _identifier_key(row, "track_index", track_index)
            selected_track_indices.add(index_key)
            if track_id is None:
                fallback_track_indices.add(index_key)

    def is_selected(row: pd.Series) -> bool:
        track_id = _IMPL._optional_int(row.get("track_id"))
        track_index = _IMPL._optional_int(row.get("track_index"))
        if track_id is not None:
            id_key = _identifier_key(row, "track_id", track_id)
            if id_key in selected_track_ids:
                return True
        if track_index is None:
            return False
        index_key = _identifier_key(row, "track_index", track_index)
        if track_id is None:
            return index_key in selected_track_indices
        return index_key in fallback_track_indices

    return np.fromiter(
        (is_selected(row) for _, row in features.iterrows()),
        dtype=bool,
        count=len(features),
    )


_IMPL._candidate_match_key = _candidate_match_key
_IMPL._selection_mask = _selection_mask

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_identifier_key"] = _identifier_key
globals()["_candidate_match_key"] = _candidate_match_key
globals()["_selection_mask"] = _selection_mask

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
