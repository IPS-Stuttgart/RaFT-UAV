"""Temporal Fortem track-level features for radar association."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int


_TRACK_SORT_COLUMNS = ("time_s", "frame_index", "track_index")
_SEQUENCE_SCOPE_COLUMNS = ("sequence_id", "flight_id")


def _row_norms(values: np.ndarray) -> np.ndarray:
    """Return row-wise Euclidean norms without losing representable finite values."""

    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(values, axis=1)
    repair = np.isfinite(values).all(axis=1) & ~np.isfinite(norms)
    if np.any(repair):
        norms = norms.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            norms[repair] = np.hypot.reduce(values[repair], axis=1)
    return norms


def _sort_track_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows in stable numeric chronology without changing their labels."""

    sort_columns = [column for column in _TRACK_SORT_COLUMNS if column in rows.columns]
    if not sort_columns:
        return rows

    sort_keys: dict[str, list[float | None]] = {}
    key_columns: list[str] = []
    for position, column in enumerate(sort_columns):
        key = f"sort_key_{position}"
        sort_keys[key] = [optional_float(value) for value in rows[column].tolist()]
        key_columns.append(key)

    row_order = (
        pd.DataFrame(sort_keys)
        .sort_values(
            key_columns,
            kind="mergesort",
            na_position="last",
        )
        .index.to_numpy()
    )
    return rows.iloc[row_order]


def add_track_level_features(radar: pd.DataFrame, *, window_frames: int = 10) -> pd.DataFrame:
    """Append causal track-level features to normalized radar rows.

    When sequence metadata is present, temporal history is scoped by every
    available ``sequence_id`` / ``flight_id`` field because Fortem track
    identifiers may be reused between independent flights.
    """

    if radar.empty or "track_id" not in radar.columns:
        return radar.copy()
    normalized_window_frames = optional_int(window_frames)
    if normalized_window_frames is None or normalized_window_frames < 1:
        raise ValueError("window_frames must be a positive integer")
    window_frames = normalized_window_frames

    original_index = radar.index.copy()
    out = _sort_track_rows(radar.reset_index(drop=True).copy())
    feature_frames: list[pd.DataFrame] = []
    known_track_ids = out["track_id"].notna()
    known_tracks = out.loc[known_track_ids]
    sequence_columns = [
        column for column in _SEQUENCE_SCOPE_COLUMNS if column in out.columns
    ]
    if sequence_columns:
        track_groups = known_tracks.groupby(
            [*sequence_columns, "track_id"],
            sort=False,
            dropna=False,
            observed=True,
        )
    else:
        track_groups = known_tracks.groupby("track_id", sort=False)
    for _, group in track_groups:
        feature_frames.append(_features_for_track(group.copy(), window_frames=window_frames))
    for row_index in out.index[~known_track_ids]:
        feature_frames.append(
            _features_for_track(out.loc[[row_index]].copy(), window_frames=window_frames)
        )
    featured = pd.concat(feature_frames, ignore_index=False).sort_index()
    featured.index = original_index
    return featured


def _features_for_track(group: pd.DataFrame, *, window_frames: int) -> pd.DataFrame:
    group = _sort_track_rows(group)
    group["track_age_frames"] = _track_age(group)
    group["track_hit_streak_frames"] = _hit_streak(group)
    group["track_time_since_first_s"] = _time_since_first(group)
    group["track_frame_gap"] = _frame_gap(group)
    group["track_position_step_m"] = _position_step(group)
    group["track_speed_from_positions_mps"] = _speed_from_positions(group)
    group["track_range_rate_mps"] = _range_rate(group)
    if "cat_prob_uav" in group.columns:
        cat = pd.to_numeric(group["cat_prob_uav"], errors="coerce")
        group["track_catprob_mean_window"] = cat.rolling(
            window_frames,
            min_periods=1,
        ).mean().to_numpy(dtype=float)
        group["track_catprob_min_window"] = cat.rolling(
            window_frames,
            min_periods=1,
        ).min().to_numpy(dtype=float)
    else:
        group["track_catprob_mean_window"] = np.nan
        group["track_catprob_min_window"] = np.nan
    group["track_velocity_smoothness_mps"] = _velocity_smoothness(
        group,
        window_frames=window_frames,
    )
    return group


def _track_age(group: pd.DataFrame) -> np.ndarray:
    """Count distinct observed frames without double-counting repeated rows."""

    if len(group) == 0:
        return np.asarray([], dtype=float)
    if "frame_index" not in group.columns:
        return np.arange(len(group), dtype=float)

    frame_index = pd.to_numeric(group["frame_index"], errors="coerce").to_numpy(dtype=float)
    age = np.zeros(len(group), dtype=float)
    for i in range(1, len(group)):
        previous = frame_index[i - 1]
        current = frame_index[i]
        same_frame = (
            np.isfinite(previous)
            and np.isfinite(current)
            and np.isclose(current, previous, rtol=0.0, atol=1.0e-9)
        )
        age[i] = age[i - 1] if same_frame else age[i - 1] + 1.0
    return age


def _hit_streak(group: pd.DataFrame) -> np.ndarray:
    if "frame_index" not in group.columns:
        return np.arange(1, len(group) + 1, dtype=float)
    frame_index = pd.to_numeric(group["frame_index"], errors="coerce").to_numpy(dtype=float)
    streak = np.ones(len(group), dtype=float)
    for i in range(1, len(group)):
        previous = frame_index[i - 1]
        current = frame_index[i]
        if not (np.isfinite(previous) and np.isfinite(current)):
            continue
        frame_delta = current - previous
        if np.isclose(frame_delta, 0.0, rtol=0.0, atol=1.0e-9):
            streak[i] = streak[i - 1]
        elif 0.0 < frame_delta <= 1.5:
            streak[i] = streak[i - 1] + 1.0
    return streak


def _time_since_first(group: pd.DataFrame) -> np.ndarray:
    times = pd.to_numeric(
        group.get("time_s", pd.Series(np.nan, index=group.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    if times.size == 0 or not np.isfinite(times[0]):
        return np.full(len(group), np.nan)
    return times - times[0]


def _frame_gap(group: pd.DataFrame) -> np.ndarray:
    if "frame_index" in group.columns:
        values = pd.to_numeric(group["frame_index"], errors="coerce").to_numpy(dtype=float)
    else:
        values = pd.to_numeric(
            group.get("time_s", pd.Series(np.nan, index=group.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
    gaps = np.r_[0.0, np.diff(values)]
    gaps = np.where(np.isfinite(gaps), gaps, np.nan)
    return np.where(gaps < 0.0, 0.0, gaps)


def _position_step(group: pd.DataFrame) -> np.ndarray:
    if not {"east_m", "north_m", "up_m"}.issubset(group.columns):
        return np.full(len(group), np.nan)
    positions = group[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    diffs = np.diff(positions, axis=0)
    steps = np.r_[0.0, _row_norms(diffs)]
    return np.where(np.isfinite(steps), steps, np.nan)


def _speed_from_positions(group: pd.DataFrame) -> np.ndarray:
    steps = _position_step(group)
    times = pd.to_numeric(
        group.get("time_s", pd.Series(np.nan, index=group.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    dt = np.r_[np.nan, np.diff(times)]
    speed = np.divide(
        steps,
        dt,
        out=np.full(len(group), np.nan),
        where=np.isfinite(dt) & (dt > 0.0),
    )
    speed[0] = np.nan
    return speed


def _range_rate(group: pd.DataFrame) -> np.ndarray:
    if "range_m" in group.columns:
        ranges = pd.to_numeric(group["range_m"], errors="coerce").to_numpy(dtype=float)
    elif {"east_m", "north_m", "up_m"}.issubset(group.columns):
        positions = group[["east_m", "north_m", "up_m"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)
        ranges = _row_norms(positions)
    else:
        return np.full(len(group), np.nan)
    times = pd.to_numeric(
        group.get("time_s", pd.Series(np.nan, index=group.index)),
        errors="coerce",
    ).to_numpy(dtype=float)
    dt = np.r_[np.nan, np.diff(times)]
    dr = np.r_[np.nan, np.diff(ranges)]
    return np.divide(
        dr,
        dt,
        out=np.full(len(group), np.nan),
        where=np.isfinite(dt) & (dt > 0.0),
    )


def _velocity_smoothness(group: pd.DataFrame, *, window_frames: int) -> np.ndarray:
    required = ["velocity_east_mps", "velocity_north_mps", "velocity_down_mps"]
    if not all(column in group.columns for column in required):
        return np.full(len(group), np.nan)
    velocity = np.column_stack(
        [
            pd.to_numeric(
                group["velocity_east_mps"],
                errors="coerce",
            ).to_numpy(dtype=float),
            pd.to_numeric(
                group["velocity_north_mps"],
                errors="coerce",
            ).to_numpy(dtype=float),
            -pd.to_numeric(
                group["velocity_down_mps"],
                errors="coerce",
            ).to_numpy(dtype=float),
        ]
    )
    diffs = np.r_[np.full((1, 3), np.nan), np.diff(velocity, axis=0)]
    norms = _row_norms(diffs)
    smoothness = (
        pd.Series(norms)
        .rolling(window_frames, min_periods=1)
        .mean()
        .to_numpy(dtype=float, copy=True)
    )
    smoothness[~np.isfinite(norms)] = np.nan
    return smoothness
