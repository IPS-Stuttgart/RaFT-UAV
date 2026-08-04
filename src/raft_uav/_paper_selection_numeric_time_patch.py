"""Normalize serialized paper-selection chronology at narrow sort boundaries."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


_paper_selection = import_module("raft_uav.paper_selection")
_PATCH_MARKER = "_raft_uav_paper_numeric_time_patch_applied"
_ORIGINAL_LARGEST_CONTINUOUS_TRACK_SEGMENT: Callable[
    [pd.DataFrame], pd.DataFrame
] = _paper_selection._largest_continuous_track_segment
_ORIGINAL_SORT_RADAR_ROWS: Callable[[pd.DataFrame], pd.DataFrame] = (
    _paper_selection._sort_radar_rows
)


def _is_missing_scalar(value: object) -> bool:
    """Return whether one cell is a scalar missing value."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _numeric_like_strings(values: pd.Series) -> pd.Series:
    """Normalize a wholly numeric serialized column and preserve other payloads."""

    materialized = values.tolist()
    nonmissing = [
        value for value in materialized if not _is_missing_scalar(value)
    ]
    if not nonmissing or not any(isinstance(value, str) for value in nonmissing):
        return values.copy()

    parsed = [optional_float(value) for value in materialized]
    for original, numeric in zip(materialized, parsed):
        if not _is_missing_scalar(original) and numeric is None:
            return values.copy()
    return pd.Series(
        [
            np.nan if _is_missing_scalar(original) else numeric
            for original, numeric in zip(materialized, parsed)
        ],
        index=values.index,
        dtype=float,
    )


def _normalize_chronology(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric serialized time/frame keys only."""

    out = frame.copy()
    for column in ("time_s", "frame_index"):
        if column in out.columns:
            out[column] = _numeric_like_strings(out[column])
    return out


def _largest_continuous_track_segment(radar: pd.DataFrame) -> pd.DataFrame:
    """Delegate segment selection after normalizing serialized chronology."""

    return _ORIGINAL_LARGEST_CONTINUOUS_TRACK_SEGMENT(
        _normalize_chronology(radar)
    )


def _sort_radar_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Delegate stable row ordering after normalizing serialized chronology."""

    return _ORIGINAL_SORT_RADAR_ROWS(_normalize_chronology(frame))


def install() -> None:
    """Install narrow chronology wrappers once per interpreter."""

    if getattr(_paper_selection, _PATCH_MARKER, False):
        return

    # Public imports resolve through the compatibility package, while its
    # maintained function objects retain the sibling legacy module as their
    # globals. Patch both namespaces so direct and internal calls share the same
    # chronology contract without replacing identifier or segmentation logic.
    _paper_selection._largest_continuous_track_segment = (
        _largest_continuous_track_segment
    )
    _paper_selection._sort_radar_rows = _sort_radar_rows
    legacy = getattr(_paper_selection, "_LEGACY", None)
    if legacy is not None:
        legacy._largest_continuous_track_segment = (
            _largest_continuous_track_segment
        )
        legacy._sort_radar_rows = _sort_radar_rows
    setattr(_paper_selection, _PATCH_MARKER, True)
