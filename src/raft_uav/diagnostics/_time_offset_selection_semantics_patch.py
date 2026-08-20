"""Make time-offset selections frame-aware and correction-safe."""

from __future__ import annotations

from functools import wraps
from importlib import import_module

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int


_time_offset = import_module("raft_uav.diagnostics.time_offset")
_PATCH_MARKER = "_raft_uav_time_offset_selection_semantics_patch_applied"
_ORIGINAL_LONGEST_TRACK_ID = _time_offset._longest_track_id
_ORIGINAL_BEST_OFFSET_ROW = _time_offset.best_offset_row


@wraps(_ORIGINAL_LONGEST_TRACK_ID)
def _longest_track_id(radar: pd.DataFrame) -> int | None:
    """Return the exact track identifier spanning the most physical frames."""

    if radar.empty or "track_id" not in radar.columns:
        return None
    if "time_s" not in radar.columns and "frame_index" not in radar.columns:
        return _ORIGINAL_LONGEST_TRACK_ID(radar)

    frame_counts: dict[int, int] = {}
    for frame in _time_offset.radar_frame_groups(radar):
        seen_in_frame: set[int] = set()
        for value in frame["track_id"]:
            track_id = optional_int(value)
            if track_id is None or track_id in seen_in_frame:
                continue
            seen_in_frame.add(track_id)
            frame_counts[track_id] = frame_counts.get(track_id, 0) + 1

    if not frame_counts:
        return None
    return int(max(frame_counts, key=frame_counts.__getitem__))


@wraps(_ORIGINAL_BEST_OFFSET_ROW)
def best_offset_row(sweep: pd.DataFrame, *, objective: str) -> pd.Series:
    """Select the best finite offset, preferring the smallest correction on ties."""

    column = _time_offset.OBJECTIVE_COLUMNS[objective]
    values = pd.to_numeric(sweep[column], errors="coerce").to_numpy(dtype=float)
    offsets = pd.to_numeric(sweep["tau_s"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values) & np.isfinite(offsets)
    if not finite.any():
        raise RuntimeError(f"no finite {column} values in offset sweep")

    candidate_positions = np.flatnonzero(finite)
    candidate_values = values[candidate_positions]
    best_value = float(np.min(candidate_values))
    tied_positions = candidate_positions[candidate_values == best_value]
    tied_offsets = offsets[tied_positions]
    tie_order = np.lexsort((tied_offsets, np.abs(tied_offsets)))
    return sweep.iloc[int(tied_positions[int(tie_order[0])])]


def install() -> None:
    """Install the time-offset selection corrections once per interpreter."""

    if getattr(_time_offset, _PATCH_MARKER, False):
        return
    _time_offset._longest_track_id = _longest_track_id
    _time_offset.best_offset_row = best_offset_row
    legacy = getattr(_time_offset, "_legacy", None)
    if legacy is not None:
        legacy._longest_track_id = _longest_track_id
        legacy.best_offset_row = best_offset_row
    setattr(_time_offset, _PATCH_MARKER, True)


install()
