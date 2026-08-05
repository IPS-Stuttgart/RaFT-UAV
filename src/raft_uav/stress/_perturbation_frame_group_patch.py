"""Keep synthetic false tracks scoped to physical radar frames."""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import perturbations as _IMPL


def _false_track_group_columns(frame: pd.DataFrame) -> list[str]:
    """Return sequence-local columns that identify one physical radar frame."""

    columns: list[str] = []
    if "sequence_id" in frame.columns:
        columns.append("sequence_id")
    if "frame_index" in frame.columns and "time_s" in frame.columns:
        columns.extend(("frame_index", "time_s"))
    elif "frame_index" in frame.columns:
        columns.append("frame_index")
    else:
        columns.append("time_s")
    return columns


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
    group_columns = _false_track_group_columns(frame)
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


_IMPL.inject_false_tracks = inject_false_tracks
