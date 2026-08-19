"""Make delayed-initialization track evidence invariant to duplicate radar rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int as _optional_int

_POSITION_COLUMNS = ("east_m", "north_m", "up_m")
_PATCH_MARKER = "_raft_uav_delayed_initialization_duplicate_track_patch_applied"


def _matching_finite_track_samples(row: pd.Series, frame: pd.DataFrame) -> pd.DataFrame:
    """Return finite same-track samples with normalized timestamps and positions."""

    track_id = _optional_int(row.get("track_id"))
    required = {*_POSITION_COLUMNS, "time_s", "track_id"}
    if track_id is None or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["time_s", *_POSITION_COLUMNS])

    matches = frame["track_id"].map(_optional_int).eq(track_id)
    track = frame.loc[matches, ["time_s", *_POSITION_COLUMNS]].copy()
    track["time_s"] = pd.to_numeric(track["time_s"], errors="coerce")
    for column in _POSITION_COLUMNS:
        track[column] = pd.to_numeric(track[column], errors="coerce")

    values = track[["time_s", *_POSITION_COLUMNS]].to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    finite = np.isfinite(values).all(axis=1)
    return track.loc[finite].copy()


def _velocity_from_track(row: pd.Series, frame: pd.DataFrame) -> np.ndarray | None:
    """Estimate endpoint velocity after collapsing co-temporal duplicate detections."""

    track = _matching_finite_track_samples(row, frame)
    if len(track) < 2:
        return None

    # Multiple rows for the same Fortem track at one timestamp are one physical
    # observation time. Averaging those positions makes the endpoint estimate
    # independent of input row order and prevents zero-duration duplicates from
    # deciding which position becomes the finite-difference endpoint.
    samples = (
        track.groupby("time_s", as_index=False, sort=True)[list(_POSITION_COLUMNS)]
        .mean()
        .sort_values("time_s", kind="stable")
    )
    if len(samples) < 2:
        return None

    times = samples["time_s"].to_numpy(dtype=float)
    positions = samples.loc[:, _POSITION_COLUMNS].to_numpy(dtype=float)
    dt = float(times[-1] - times[0])
    if dt <= 0.0:
        return None
    velocity = (positions[-1] - positions[0]) / dt
    return velocity if np.isfinite(velocity).all() else None


def _track_support_score(row: pd.Series, radar: pd.DataFrame) -> float:
    """Score temporal track support by distinct finite observation timestamps."""

    track_id = _optional_int(row.get("track_id"))
    if track_id is None or not {"track_id", "time_s"}.issubset(radar.columns):
        return 1.0

    matches = radar["track_id"].map(_optional_int).eq(track_id)
    times = pd.to_numeric(radar.loc[matches, "time_s"], errors="coerce")
    time_values = times.to_numpy(dtype=float, na_value=np.nan)
    finite_times = time_values[np.isfinite(time_values)]
    count = int(np.unique(finite_times).size)
    return float(1.0 / max(count, 1))


def apply_delayed_initialization_duplicate_track_patch(delayed_module: Any) -> None:
    """Install duplicate-safe delayed-initialization track evidence helpers."""

    if getattr(delayed_module, _PATCH_MARKER, False):
        return

    delayed_module._velocity_from_track = _velocity_from_track
    delayed_module._track_support_score = _track_support_score
    setattr(delayed_module, _PATCH_MARKER, True)
