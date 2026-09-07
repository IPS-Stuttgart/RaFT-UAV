"""Keep finite nonnegative tracklet summaries stable during mean reduction."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_tracklet_nonnegative_mean_stability"


def _stable_nanmean_nonnegative(values: Any) -> float:
    """Return a NaN-skipping mean without overflowing a finite positive sum."""

    array = np.asarray(values, dtype=float).reshape(-1)
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return float("nan")
    maximum = float(np.max(valid))
    if maximum == 0.0 or not np.isfinite(maximum):
        return maximum
    with np.errstate(under="ignore"):
        scaled = valid / maximum
        mean_scaled = float(np.mean(scaled))
    return float(maximum * mean_scaled)


def _stable_tracklet_means(
    implementation: ModuleType,
    segment: pd.DataFrame,
) -> tuple[float, float]:
    """Recompute the two nonnegative tracklet means without sum overflow."""

    times = pd.to_numeric(segment["time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    positions = segment.loc[:, implementation.PositionColumns].to_numpy(
        dtype=float
    )
    with np.errstate(over="ignore", invalid="ignore"):
        dt = np.diff(times)
    if len(segment) > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            position_deltas = np.diff(positions, axis=0)
        displacement = implementation._euclidean_norm(position_deltas, axis=1)
    else:
        displacement = np.empty(0)

    valid_speed_intervals = np.isfinite(dt) & (dt > 1.0e-9)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        speeds = np.divide(
            displacement,
            dt,
            out=np.full_like(displacement, np.nan, dtype=float),
            where=valid_speed_intervals,
        )
    finite_endpoints = np.isfinite(positions[:-1]).all(axis=1) & np.isfinite(
        positions[1:]
    ).all(axis=1)
    repair = valid_speed_intervals & finite_endpoints & ~np.isfinite(speeds)
    for interval_index in np.flatnonzero(repair):
        speeds[interval_index] = implementation.stable_euclidean_rate(
            positions[interval_index + 1],
            positions[interval_index],
            float(dt[interval_index]),
        )

    finite_speeds = speeds[np.isfinite(speeds)]
    mean_speed = (
        _stable_nanmean_nonnegative(finite_speeds)
        if finite_speeds.size
        else (0.0 if speeds.size == 0 else float("nan"))
    )
    ranges = implementation._euclidean_norm(positions, axis=1)
    mean_range = _stable_nanmean_nonnegative(ranges)
    return mean_speed, mean_range


def _wrap_tracklet_features(
    implementation: ModuleType,
    original: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """Replace only overflow-prone nonnegative means in tracklet summaries."""

    @wraps(original)
    def wrapped(
        segment: pd.DataFrame,
        track_id: object,
        segment_index: int,
    ) -> dict[str, object]:
        with np.errstate(over="ignore"):
            result = original(segment, track_id, segment_index)
        mean_speed, mean_range = _stable_tracklet_means(implementation, segment)
        result["mean_speed_mps"] = mean_speed
        result["mean_range_m"] = mean_range
        return result

    return wrapped


def _wrap_frame_context_features(
    implementation: ModuleType,
    original: Callable[[pd.DataFrame], pd.DataFrame],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Repair row-wise neighbor means after preserving all other features."""

    @wraps(original)
    def wrapped(candidates: pd.DataFrame) -> pd.DataFrame:
        with np.errstate(over="ignore"):
            out = original(candidates)
        if len(out) <= 1:
            return out

        positions = candidates.loc[:, implementation.PositionColumns].to_numpy(
            dtype=float
        )
        with np.errstate(over="ignore", invalid="ignore"):
            deltas = positions[:, None, :] - positions[None, :, :]
        distances = implementation._euclidean_norm(deltas, axis=2)
        np.fill_diagonal(distances, np.nan)
        out["mean_neighbor_distance_m"] = np.asarray(
            [_stable_nanmean_nonnegative(row) for row in distances],
            dtype=float,
        )
        return out

    return wrapped


def apply_tracklet_nonnegative_mean_stability_patch(module: ModuleType) -> None:
    """Install overflow-stable mean summaries on the maintained tracklet API."""

    implementation = getattr(module, "_IMPL", module)
    if getattr(implementation, _PATCH_MARKER, False):
        return

    tracklet_features = _wrap_tracklet_features(
        implementation,
        implementation._tracklet_features,
    )
    frame_context_features = _wrap_frame_context_features(
        implementation,
        implementation.frame_context_features,
    )
    implementation._tracklet_features = tracklet_features
    implementation.frame_context_features = frame_context_features
    module._tracklet_features = tracklet_features
    module.frame_context_features = frame_context_features
    setattr(implementation, _PATCH_MARKER, True)
    setattr(module, _PATCH_MARKER, True)
