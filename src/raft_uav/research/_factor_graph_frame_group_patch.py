"""Keep reused radar frame counters separate in factor-graph association."""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_groups_factor_graph_frames_by_index_and_time"
_SORT_KEY_PREFIX = "_raft_uav_factor_graph_sort_"


def _finite_real_scalar(value: object) -> float | None:
    """Return a finite real scalar without complex-column dtype poisoning."""

    if value is None or np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.ndim != 0:
        return None
    scalar = array.item()
    if np.ma.is_masked(scalar) or isinstance(scalar, (bool, np.bool_)):
        return None
    if isinstance(scalar, (complex, np.complexfloating)):
        real = float(np.real(scalar))
        imaginary = float(np.imag(scalar))
        if not np.isfinite(real) or not np.isfinite(imaginary) or imaginary != 0.0:
            return None
        return real
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _finite_real_series(values: pd.Series) -> pd.Series:
    """Normalize scalar-like values independently while preserving their index."""

    return pd.Series(
        [_finite_real_scalar(value) for value in values],
        index=values.index,
        dtype=float,
    )


def _numeric_track_id_sort_keys(values: pd.Series) -> pd.DataFrame:
    """Return deterministic numeric-first track-ID ordering keys."""

    numeric = _finite_real_series(values)
    return pd.DataFrame(
        {
            f"{_SORT_KEY_PREFIX}track_id_is_text": numeric.isna(),
            f"{_SORT_KEY_PREFIX}track_id_numeric": numeric,
            f"{_SORT_KEY_PREFIX}track_id_text": values.where(
                values.notna(), ""
            ).astype(str),
        },
        index=values.index,
    )


def _ordered_radar_rows(radar: pd.DataFrame) -> pd.DataFrame:
    """Sort frame keys numerically without mutating the radar payload."""

    rows = pd.DataFrame(radar).copy()
    if rows.empty:
        return rows.reset_index(drop=True)

    times = (
        _finite_real_series(rows["time_s"])
        if "time_s" in rows.columns
        else pd.Series(np.nan, index=rows.index, dtype=float)
    )
    frame_indices = (
        _finite_real_series(rows["frame_index"])
        if "frame_index" in rows.columns
        else pd.Series(np.nan, index=rows.index, dtype=float)
    )
    # A valid timestamp is the primary chronology key. If it is malformed, fall
    # back to the frame index so one bad time value cannot move an otherwise valid
    # frame to the end of the association sequence.
    sort_keys = pd.DataFrame(
        {
            f"{_SORT_KEY_PREFIX}chronology": times.where(
                times.notna(), frame_indices
            ),
            f"{_SORT_KEY_PREFIX}frame_index": frame_indices.where(
                frame_indices.notna(), times
            ),
        },
        index=rows.index,
    )
    sort_columns = list(sort_keys.columns)

    if "track_id" in rows.columns:
        track_id_keys = _numeric_track_id_sort_keys(rows["track_id"])
        sort_keys = pd.concat([sort_keys, track_id_keys], axis=1)
        sort_columns.extend(track_id_keys.columns)

    row_order = sort_keys.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="last",
    ).index
    return rows.loc[row_order].reset_index(drop=True)


def apply_factor_graph_frame_group_patch(module: ModuleType) -> None:
    """Patch factor-graph radar grouping to disambiguate counter reuse."""

    implementation = getattr(module, "_LEGACY", module)
    original = implementation._radar_frame_groups
    if getattr(original, _PATCH_MARKER, False):
        module._radar_frame_groups = original
        return

    @wraps(original)
    def radar_frame_groups(
        radar: pd.DataFrame,
    ) -> list[tuple[object, pd.DataFrame]]:
        ordered = _ordered_radar_rows(radar)
        times = (
            _finite_real_series(ordered["time_s"])
            if "time_s" in ordered.columns
            else pd.Series(np.nan, index=ordered.index, dtype=float)
        )
        frame_indices = (
            _finite_real_series(ordered["frame_index"])
            if "frame_index" in ordered.columns
            else pd.Series(np.nan, index=ordered.index, dtype=float)
        )

        group_keys = pd.Series(
            [
                ("frame_index_time", float(frame_index), float(time_s))
                if np.isfinite(frame_index) and np.isfinite(time_s)
                else ("frame_index", float(frame_index))
                if np.isfinite(frame_index)
                else ("time_s", float(time_s))
                if np.isfinite(time_s)
                else None
                for frame_index, time_s in zip(frame_indices, times, strict=True)
            ],
            index=ordered.index,
            dtype=object,
        )
        usable = group_keys.notna()
        ordered = ordered.loc[usable]
        group_keys = group_keys.loc[usable]
        return [
            (key, group.copy())
            for key, group in ordered.groupby(group_keys, sort=False)
        ]

    setattr(radar_frame_groups, _PATCH_MARKER, True)
    implementation._radar_frame_groups = radar_frame_groups
    module._radar_frame_groups = radar_frame_groups
