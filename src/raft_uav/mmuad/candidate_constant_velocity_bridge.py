"""Truth-free constant-velocity bridge diagnostics for MMUAD candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns


@dataclass(frozen=True)
class ConstantVelocityBridgeConfig:
    """Controls for bracketing constant-velocity support."""

    bridge_top_n: int = 2
    max_frame_gap_s: float = 1.0
    max_speed_mps: float = 60.0
    max_interpolation_error_m: float = 5.0
    interpolation_scale_m: float = 5.0
    max_neighbors_per_side: int = 40
    require_same_source: bool = False
    require_same_branch: bool = False
    protect_bridge_quota: bool = True
    reason_prefix: str = "cv_bridge"


def attach_constant_velocity_bridge_features(
    candidates: pd.DataFrame,
    *,
    config: ConstantVelocityBridgeConfig | None = None,
) -> pd.DataFrame:
    """Attach bridge support from the nearest earlier and later candidate frames."""

    config = config or ConstantVelocityBridgeConfig()
    validate_constant_velocity_bridge_config(config)
    raw = pd.DataFrame(candidates).copy()
    raw["_cv_bridge_input_row"] = np.arange(len(raw), dtype=int)
    rows = normalize_candidate_columns(raw)
    if rows.empty:
        return _empty_columns(rows)
    rows = rows.copy().reset_index(drop=True)
    rows["source"] = rows.get("source", "unknown").fillna("unknown").astype(str)
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"]
    rows["candidate_branch"] = rows["candidate_branch"].fillna("candidate").astype(str)

    parts = [
        _attach_sequence(group, config=config)
        for _, group in rows.groupby("sequence_id", sort=False, dropna=False)
    ]
    out = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    return out.sort_values("_cv_bridge_input_row").reset_index(drop=True)


def validate_constant_velocity_bridge_config(config: ConstantVelocityBridgeConfig) -> None:
    """Reject invalid controls before pair construction."""

    positive = {
        "max_frame_gap_s": config.max_frame_gap_s,
        "max_speed_mps": config.max_speed_mps,
        "max_interpolation_error_m": config.max_interpolation_error_m,
        "interpolation_scale_m": config.interpolation_scale_m,
    }
    for name, value in positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if int(config.bridge_top_n) < 0:
        raise ValueError("bridge_top_n must be non-negative")
    if int(config.max_neighbors_per_side) < 0:
        raise ValueError("max_neighbors_per_side must be non-negative")
    if not str(config.reason_prefix).strip():
        raise ValueError("reason_prefix must be non-empty")


def _attach_sequence(
    rows: pd.DataFrame,
    *,
    config: ConstantVelocityBridgeConfig,
) -> pd.DataFrame:
    times = np.sort(pd.to_numeric(rows["time_s"], errors="coerce").dropna().unique())
    by_time = {
        float(time_s): group.copy()
        for time_s, group in rows.groupby("time_s", sort=False, dropna=False)
        if np.isfinite(float(time_s))
    }
    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        time_s = float(row["time_s"])
        previous_time, next_time = _adjacent_times(times, time_s)
        result = _best_bridge(
            row,
            by_time.get(previous_time),
            by_time.get(next_time),
            previous_dt_s=None if previous_time is None else time_s - previous_time,
            next_dt_s=None if next_time is None else next_time - time_s,
            config=config,
        )
        record = row.to_dict()
        record.update(result)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _adjacent_times(times: np.ndarray, time_s: float) -> tuple[float | None, float | None]:
    left = int(np.searchsorted(times, time_s, side="left"))
    right = int(np.searchsorted(times, time_s, side="right"))
    previous = float(times[left - 1]) if left > 0 else None
    following = float(times[right]) if right < len(times) else None
    return previous, following


def _best_bridge(
    row: pd.Series,
    previous: pd.DataFrame | None,
    following: pd.DataFrame | None,
    *,
    previous_dt_s: float | None,
    next_dt_s: float | None,
    config: ConstantVelocityBridgeConfig,
) -> dict[str, Any]:
    result = _empty_result(previous_dt_s=previous_dt_s, next_dt_s=next_dt_s)
    if previous is None or following is None or previous.empty or following.empty:
        return result
    if previous_dt_s is None or next_dt_s is None:
        return result
    previous_dt = float(previous_dt_s)
    next_dt = float(next_dt_s)
    if not _valid_gap(previous_dt, config.max_frame_gap_s):
        return result
    if not _valid_gap(next_dt, config.max_frame_gap_s):
        return result

    previous = _compatible(previous, row=row, config=config)
    following = _compatible(following, row=row, config=config)
    if previous.empty or following.empty:
        return result
    center = row[["x_m", "y_m", "z_m"]].to_numpy(float)
    previous = _nearest(previous, center=center, count=config.max_neighbors_per_side)
    following = _nearest(following, center=center, count=config.max_neighbors_per_side)

    previous_xyz = previous[["x_m", "y_m", "z_m"]].to_numpy(float)
    following_xyz = following[["x_m", "y_m", "z_m"]].to_numpy(float)
    previous_speed = np.linalg.norm(previous_xyz - center, axis=1) / previous_dt
    next_speed = np.linalg.norm(following_xyz - center, axis=1) / next_dt
    total_dt = previous_dt + next_dt
    segment_delta = following_xyz[None, :, :] - previous_xyz[:, None, :]
    segment_speed = np.linalg.norm(segment_delta, axis=2) / total_dt
    predicted = previous_xyz[:, None, :] + (previous_dt / total_dt) * segment_delta
    error = np.linalg.norm(predicted - center[None, None, :], axis=2)
    valid = (
        (previous_speed[:, None] <= config.max_speed_mps)
        & (next_speed[None, :] <= config.max_speed_mps)
        & (segment_speed <= config.max_speed_mps)
        & (error <= config.max_interpolation_error_m)
    )
    if not bool(valid.any()):
        return result

    valid_error = np.where(valid, error, np.inf)
    previous_index, next_index = np.unravel_index(
        int(np.argmin(valid_error)),
        valid_error.shape,
    )
    best_error = float(error[previous_index, next_index])
    result.update(
        {
            "candidate_cv_bridge_supported": True,
            "candidate_cv_bridge_error_m": best_error,
            "candidate_cv_bridge_score": float(
                np.exp(-best_error / config.interpolation_scale_m)
            ),
            "candidate_cv_bridge_prev_speed_mps": float(previous_speed[previous_index]),
            "candidate_cv_bridge_next_speed_mps": float(next_speed[next_index]),
            "candidate_cv_bridge_segment_speed_mps": float(
                segment_speed[previous_index, next_index]
            ),
        }
    )
    return result


def _compatible(
    rows: pd.DataFrame,
    *,
    row: pd.Series,
    config: ConstantVelocityBridgeConfig,
) -> pd.DataFrame:
    out = rows.copy()
    if config.require_same_source:
        out = out.loc[out["source"].astype(str) == str(row["source"])]
    if config.require_same_branch:
        out = out.loc[
            out["candidate_branch"].astype(str) == str(row["candidate_branch"])
        ]
    return out


def _nearest(rows: pd.DataFrame, *, center: np.ndarray, count: int) -> pd.DataFrame:
    if count <= 0 or len(rows) <= count:
        return rows
    xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    order = np.argsort(np.linalg.norm(xyz - center, axis=1), kind="stable")[:count]
    return rows.iloc[order]


def _valid_gap(value: float, maximum: float) -> bool:
    return bool(np.isfinite(value) and 0.0 < value <= maximum)


def _empty_result(
    *,
    previous_dt_s: float | None,
    next_dt_s: float | None,
) -> dict[str, Any]:
    return {
        "candidate_cv_bridge_supported": False,
        "candidate_cv_bridge_error_m": float("nan"),
        "candidate_cv_bridge_score": 0.0,
        "candidate_cv_bridge_prev_speed_mps": float("nan"),
        "candidate_cv_bridge_next_speed_mps": float("nan"),
        "candidate_cv_bridge_segment_speed_mps": float("nan"),
        "candidate_cv_bridge_prev_dt_s": (
            float("nan") if previous_dt_s is None else float(previous_dt_s)
        ),
        "candidate_cv_bridge_next_dt_s": (
            float("nan") if next_dt_s is None else float(next_dt_s)
        ),
    }


def _empty_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["candidate_cv_bridge_supported"] = pd.Series(dtype=bool)
    for column in (
        "candidate_cv_bridge_error_m",
        "candidate_cv_bridge_score",
        "candidate_cv_bridge_prev_speed_mps",
        "candidate_cv_bridge_next_speed_mps",
        "candidate_cv_bridge_segment_speed_mps",
        "candidate_cv_bridge_prev_dt_s",
        "candidate_cv_bridge_next_dt_s",
    ):
        out[column] = pd.Series(dtype=float)
    return out
