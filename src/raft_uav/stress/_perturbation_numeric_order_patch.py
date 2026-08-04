"""Sort stress-test outputs by numeric-like chronology keys."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

from . import perturbations as _IMPL


_NUMERIC_SORT_COLUMNS = frozenset({"time_s", "frame_index", "track_id"})


def _wholly_numeric_sort_key(values: pd.Series) -> pd.Series | None:
    """Return a finite numeric key when every present value is numeric-like."""

    numeric = values.map(optional_float)
    present = values.notna()
    if bool(present.any()) and not bool(numeric.loc[present].notna().all()):
        return None
    return numeric.astype(float)


def _temporary_column_name(frame: pd.DataFrame, column: str, index: int) -> str:
    candidate = f"_stress_numeric_sort_{index}_{column}"
    while candidate in frame.columns:
        candidate = f"_{candidate}"
    return candidate


def _sort_stress_output(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Sort numeric-like keys numerically without mutating payload columns."""

    out = frame.copy()
    sort_columns: list[str] = []
    temporary_columns: list[str] = []
    for index, column in enumerate(columns):
        if column not in out.columns:
            continue
        if column not in _NUMERIC_SORT_COLUMNS:
            sort_columns.append(column)
            continue
        numeric_key = _wholly_numeric_sort_key(out[column])
        if numeric_key is None:
            sort_columns.append(column)
            continue
        temporary_column = _temporary_column_name(out, column, index)
        out[temporary_column] = numeric_key
        sort_columns.append(temporary_column)
        temporary_columns.append(temporary_column)

    if not sort_columns:
        return out
    sorted_frame = out.sort_values(sort_columns, kind="mergesort")
    return sorted_frame.drop(columns=temporary_columns)


def perturb_radar(
    radar: pd.DataFrame,
    config: _IMPL.PerturbationConfig,
) -> pd.DataFrame:
    """Return perturbed radar rows in numeric chronological order."""

    rng = np.random.default_rng(config.seed)
    out = radar.copy()
    out = _IMPL.drop_radar_frames(out, rate=config.radar_drop_rate, rng=rng)
    out = _IMPL.jitter_timestamps(out, std_s=config.timestamp_jitter_std_s, rng=rng)
    out = _IMPL.corrupt_velocity(out, std_mps=config.velocity_noise_std_mps, rng=rng)
    out = _IMPL.scale_covariance_columns(out, scale=config.covariance_scale)
    out = _IMPL.inject_false_tracks(
        out,
        false_tracks_per_frame=config.false_tracks_per_frame,
        position_std_m=config.false_track_position_std_m,
        rng=rng,
    )
    out["stress_config"] = config.name
    return _sort_stress_output(out, ("time_s", "frame_index", "track_id"))


def perturb_rf(
    rf: pd.DataFrame,
    config: _IMPL.PerturbationConfig,
) -> pd.DataFrame:
    """Return perturbed RF rows in numeric chronological order."""

    rng = np.random.default_rng(config.seed + 17)
    out = rf.copy()
    out = _IMPL.drop_rf_bursts(out, rate=config.rf_drop_burst_rate, rng=rng)
    out = _IMPL.jitter_timestamps(out, std_s=config.timestamp_jitter_std_s, rng=rng)
    out = _IMPL.scale_covariance_columns(out, scale=config.covariance_scale)
    out["stress_config"] = config.name
    return _sort_stress_output(out, ("time_s",))


_IMPL.perturb_radar = perturb_radar
_IMPL.perturb_rf = perturb_rf
