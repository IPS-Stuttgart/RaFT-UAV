"""Deterministic stress-test perturbations for normalized RF/radar frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for one stress-test perturbation set."""

    name: str
    radar_drop_rate: float = 0.0
    rf_drop_burst_rate: float = 0.0
    timestamp_jitter_std_s: float = 0.0
    false_tracks_per_frame: int = 0
    false_track_position_std_m: float = 150.0
    velocity_noise_std_mps: float = 0.0
    covariance_scale: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        for field_name in ("radar_drop_rate", "rf_drop_burst_rate"):
            object.__setattr__(
                self,
                field_name,
                _drop_rate(getattr(self, field_name), name=field_name),
            )
        for field_name in (
            "timestamp_jitter_std_s",
            "false_track_position_std_m",
            "velocity_noise_std_mps",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative_float(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "covariance_scale",
            _finite_positive_float(self.covariance_scale, name="covariance_scale"),
        )
        object.__setattr__(
            self,
            "false_tracks_per_frame",
            _nonnegative_int(self.false_tracks_per_frame, name="false_tracks_per_frame"),
        )
        object.__setattr__(self, "seed", _nonnegative_int(self.seed, name="seed"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PerturbationConfig":
        fields = cls.__dataclass_fields__
        return cls(**{field: payload[field] for field in fields if field in payload})


def perturb_radar(radar: pd.DataFrame, config: PerturbationConfig) -> pd.DataFrame:
    """Return a perturbed radar frame."""

    rng = np.random.default_rng(config.seed)
    out = radar.copy()
    out = drop_radar_frames(out, rate=config.radar_drop_rate, rng=rng)
    out = jitter_timestamps(out, std_s=config.timestamp_jitter_std_s, rng=rng)
    out = corrupt_velocity(out, std_mps=config.velocity_noise_std_mps, rng=rng)
    out = scale_covariance_columns(out, scale=config.covariance_scale)
    out = inject_false_tracks(
        out,
        false_tracks_per_frame=config.false_tracks_per_frame,
        position_std_m=config.false_track_position_std_m,
        rng=rng,
    )
    out["stress_config"] = config.name
    sort_columns = [c for c in ("time_s", "frame_index", "track_id") if c in out.columns]
    return out.sort_values(sort_columns) if sort_columns else out


def perturb_rf(rf: pd.DataFrame, config: PerturbationConfig) -> pd.DataFrame:
    """Return a perturbed RF frame."""

    rng = np.random.default_rng(config.seed + 17)
    out = rf.copy()
    out = drop_rf_bursts(out, rate=config.rf_drop_burst_rate, rng=rng)
    out = jitter_timestamps(out, std_s=config.timestamp_jitter_std_s, rng=rng)
    out = scale_covariance_columns(out, scale=config.covariance_scale)
    out["stress_config"] = config.name
    return out.sort_values("time_s") if "time_s" in out.columns else out


def drop_radar_frames(
    frame: pd.DataFrame,
    *,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    drop_rate = _drop_rate(rate, name="rate")
    if frame.empty or drop_rate == 0.0:
        return frame.copy()

    frame_column = "frame_index" if "frame_index" in frame.columns else "time_s"
    group_columns = [frame_column]
    if "sequence_id" in frame.columns:
        group_columns.insert(0, "sequence_id")

    valid_group_mask = frame[frame_column].notna().to_numpy()
    group_ids = (
        frame.loc[valid_group_mask, group_columns]
        .groupby(group_columns, sort=True, dropna=False)
        .ngroup()
        .to_numpy()
    )
    groups = pd.Series(np.unique(group_ids))
    keep_groups = set(groups.loc[rng.random(len(groups)) >= drop_rate].tolist())
    keep_mask = np.ones(len(frame), dtype=bool)
    keep_mask[valid_group_mask] = np.isin(group_ids, list(keep_groups))
    return frame.loc[keep_mask].copy()


def drop_rf_bursts(frame: pd.DataFrame, *, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    drop_rate = _drop_rate(rate, name="rate")
    if frame.empty or drop_rate == 0.0 or "time_s" not in frame.columns:
        return frame.copy()
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    if times.size == 0:
        return frame.copy()
    valid_time_mask = np.isfinite(times)
    if not np.any(valid_time_mask):
        return frame.copy()

    finite_rows = frame.loc[valid_time_mask].copy()
    finite_rows["_stress_time_s"] = times[valid_time_mask]
    group_columns = ["_stress_burst_bin"]
    if "sequence_id" in finite_rows.columns:
        group_columns.insert(0, "sequence_id")
        sequence_start = finite_rows.groupby(
            "sequence_id",
            sort=False,
            dropna=False,
        )["_stress_time_s"].transform("min")
    else:
        sequence_start = float(finite_rows["_stress_time_s"].min())
    finite_rows["_stress_burst_bin"] = np.floor(
        (finite_rows["_stress_time_s"] - sequence_start) / 5.0
    ).astype(int)
    group_ids = (
        finite_rows.groupby(group_columns, sort=True, dropna=False)
        .ngroup()
        .to_numpy()
    )
    groups = pd.Series(np.unique(group_ids))
    dropped = set(groups.loc[rng.random(len(groups)) < drop_rate].tolist())

    keep_mask = np.ones(times.shape, dtype=bool)
    keep_mask[valid_time_mask] = ~np.isin(group_ids, list(dropped))
    return frame.loc[keep_mask].copy()


def jitter_timestamps(
    frame: pd.DataFrame,
    *,
    std_s: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    jitter_std_s = _finite_nonnegative_float(std_s, name="std_s")
    out = frame.copy()
    if jitter_std_s == 0.0 or "time_s" not in out.columns:
        return out
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce") + rng.normal(
        0.0,
        jitter_std_s,
        len(out),
    )
    return out


def corrupt_velocity(
    frame: pd.DataFrame,
    *,
    std_mps: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    velocity_std_mps = _finite_nonnegative_float(std_mps, name="std_mps")
    out = frame.copy()
    if velocity_std_mps == 0.0:
        return out
    for column in ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") + rng.normal(
                0.0,
                velocity_std_mps,
                len(out),
            )
    return out


def scale_covariance_columns(frame: pd.DataFrame, *, scale: float) -> pd.DataFrame:
    out = frame.copy()
    scale_value = _finite_positive_float(scale, name="covariance scale")
    if scale_value == 1.0:
        return out
    for column in out.columns:
        if column.startswith("cov_") or column.startswith("association_cov_"):
            out[column] = pd.to_numeric(out[column], errors="coerce") * scale_value
    return out


def inject_false_tracks(
    frame: pd.DataFrame,
    *,
    false_tracks_per_frame: int,
    position_std_m: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    track_count = _nonnegative_int(false_tracks_per_frame, name="false_tracks_per_frame")
    false_position_std_m = _finite_nonnegative_float(position_std_m, name="position_std_m")
    if frame.empty or track_count == 0 or not {"east_m", "north_m", "up_m"}.issubset(frame.columns):
        return frame.copy()
    frame_column = "frame_index" if "frame_index" in frame.columns else "time_s"
    group_columns = [frame_column]
    if "sequence_id" in frame.columns:
        group_columns.insert(0, "sequence_id")
    rows: list[pd.Series] = []
    next_track_id = _next_false_track_id(frame)
    for _, group in frame.groupby(group_columns, sort=True, dropna=False):
        reference = group.iloc[0]
        center = group[["east_m", "north_m", "up_m"]].mean().to_numpy(dtype=float)
        for index in range(track_count):
            row = reference.copy()
            position = center + rng.normal(0.0, false_position_std_m, 3)
            row["east_m"], row["north_m"], row["up_m"] = [float(value) for value in position]
            row["track_id"] = next_track_id + index
            row["cat_prob_uav"] = _false_track_cat_probability(
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


def _false_track_cat_probability(value: object) -> float:
    """Return a finite low-confidence probability for a synthetic false track."""

    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.2
    if not np.isfinite(probability):
        return 0.2
    return float(np.clip(probability, 0.0, 0.2))


def _drop_rate(value: object, *, name: str) -> float:
    """Return a finite probability without silently saturating values above one."""

    rate = _finite_nonnegative_float(value, name=name)
    if rate > 1.0:
        raise ValueError(f"{name} must not exceed 1")
    return rate


def _finite_nonnegative_float(value: object, *, name: str) -> float:
    number = optional_float(value)
    if number is None or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _finite_positive_float(value: object, *, name: str) -> float:
    number = optional_float(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative_int(value: object, *, name: str) -> int:
    number = optional_int(value)
    if number is None or number < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return number


def _next_false_track_id(frame: pd.DataFrame) -> int:
    if "track_id" not in frame.columns:
        return 10_000_000
    values = pd.to_numeric(frame["track_id"], errors="coerce").dropna()
    return 10_000_000 if values.empty else int(values.max()) + 1
