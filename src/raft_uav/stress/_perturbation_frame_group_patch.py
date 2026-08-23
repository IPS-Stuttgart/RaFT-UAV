"""Keep radar stress perturbations scoped to physical radar frames."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import perturbations as _IMPL

_SCOPE_COLUMNS = ("sequence_id", "flight_id")


def _physical_frame_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that identify one physical radar frame."""

    columns = [column for column in _SCOPE_COLUMNS if column in frame.columns]
    if "frame_index" in frame.columns and "time_s" in frame.columns:
        columns.extend(("frame_index", "time_s"))
    elif "frame_index" in frame.columns:
        columns.append("frame_index")
    else:
        columns.append("time_s")
    return columns


def drop_radar_frames(
    frame: pd.DataFrame,
    *,
    rate: float,
    rng: Any,
) -> pd.DataFrame:
    """Drop physical radar frames independently, including reused counters."""

    drop_rate = _IMPL._drop_rate(rate, name="rate")
    if frame.empty or drop_rate == 0.0:
        return frame.copy()

    if "frame_index" in frame.columns and "time_s" in frame.columns:
        valid_group_mask = (
            frame["frame_index"].notna() | frame["time_s"].notna()
        ).to_numpy()
    else:
        frame_column = "frame_index" if "frame_index" in frame.columns else "time_s"
        valid_group_mask = frame[frame_column].notna().to_numpy()
    group_columns = _physical_frame_group_columns(frame)
    group_ids = (
        frame.loc[valid_group_mask, group_columns]
        .groupby(group_columns, sort=True, dropna=False)
        .ngroup()
        .to_numpy()
    )
    groups = np.unique(group_ids)
    keep_groups = set(groups[rng.random(len(groups)) >= drop_rate].tolist())
    keep_mask = np.ones(len(frame), dtype=bool)
    keep_mask[valid_group_mask] = np.isin(group_ids, list(keep_groups))
    return frame.loc[keep_mask].copy()


def jitter_timestamps(
    frame: pd.DataFrame,
    *,
    std_s: float,
    rng: Any,
) -> pd.DataFrame:
    """Jitter each physical radar frame once, including reused counters."""

    jitter_std_s = _IMPL._finite_nonnegative_float(std_s, name="std_s")
    out = frame.copy()
    if jitter_std_s == 0.0 or "time_s" not in out.columns or out.empty:
        return out

    times = pd.to_numeric(out["time_s"], errors="coerce")
    group_columns = _physical_frame_group_columns(out)
    group_keys = out.loc[:, group_columns].copy()
    group_keys["time_s"] = times
    group_ids = (
        group_keys.groupby(group_columns, sort=True, dropna=False)
        .ngroup()
        .to_numpy()
    )
    group_count = int(group_ids.max()) + 1
    row_jitter = rng.normal(0.0, jitter_std_s, len(out))
    first_positions = np.full(group_count, len(group_ids), dtype=int)
    np.minimum.at(first_positions, group_ids, np.arange(len(group_ids)))
    out["time_s"] = times.to_numpy() + row_jitter[first_positions[group_ids]]
    return out


def inject_false_tracks(
    frame: pd.DataFrame,
    *,
    false_tracks_per_frame: int,
    position_std_m: float,
    rng: Any,
) -> pd.DataFrame:
    """Inject false tracks once per physical frame, including reused counters."""

    track_count = _IMPL._nonnegative_int(
        false_tracks_per_frame,
        name="false_tracks_per_frame",
    )
    false_position_std_m = _IMPL._finite_nonnegative_float(
        position_std_m,
        name="position_std_m",
    )
    if (
        frame.empty
        or track_count == 0
        or not {"east_m", "north_m", "up_m"}.issubset(frame.columns)
    ):
        return frame.copy()

    rows: list[pd.Series] = []
    next_track_id = _IMPL._next_false_track_id(frame)
    group_columns = _physical_frame_group_columns(frame)
    for _, group in frame.groupby(group_columns, sort=True, dropna=False):
        reference = group.iloc[0]
        center = group[["east_m", "north_m", "up_m"]].mean().to_numpy(dtype=float)
        for index in range(track_count):
            row = reference.copy()
            position = center + rng.normal(0.0, false_position_std_m, 3)
            row["east_m"], row["north_m"], row["up_m"] = [
                float(value) for value in position
            ]
            row["track_id"] = next_track_id + index
            row["cat_prob_uav"] = _IMPL._false_track_cat_probability(
                row.get("cat_prob_uav", 0.2)
            )
            row["stress_false_track"] = True
            rows.append(row)
        next_track_id += track_count
    if not rows:
        return frame.copy()
    out = pd.concat([frame.copy(), pd.DataFrame(rows)], ignore_index=True)
    if "stress_false_track" not in out.columns:
        out["stress_false_track"] = False
    out["stress_false_track"] = out["stress_false_track"].fillna(False).astype(bool)
    return out


_IMPL.drop_radar_frames = drop_radar_frames
_IMPL.jitter_timestamps = jitter_timestamps
_IMPL.inject_false_tracks = inject_false_tracks
