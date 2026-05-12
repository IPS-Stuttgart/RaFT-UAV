"""Compact run diagnostics for tracker artifacts."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.evaluation.metrics import nearest_time_indices


def build_diagnostic_summary(
    *,
    estimate_frame: pd.DataFrame,
    selected_radar: pd.DataFrame,
    truth: pd.DataFrame,
    max_eval_time_delta_s: float | None,
    top_n: int = 20,
    window_s: float = 30.0,
) -> dict[str, Any]:
    """Build a small JSON-serializable diagnostic summary for one tracker run."""

    if top_n < 1:
        raise ValueError("top_n must be positive")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")

    return {
        "schema_version": 1,
        "top_n": int(top_n),
        "window_s": float(window_s),
        "top_residuals": _top_residuals(estimate_frame, top_n=top_n),
        "track_switches": {
            "posterior_radar": _track_switch_summary(
                estimate_frame.loc[_source_mask(estimate_frame, "radar")],
                top_n=top_n,
            ),
            "selected_radar": _track_switch_summary(selected_radar, top_n=top_n),
        },
        "covariance_inflation": _covariance_inflation_summary(estimate_frame, top_n=top_n),
        "worst_time_windows": _worst_time_windows(
            estimate_frame=estimate_frame,
            truth=truth,
            max_eval_time_delta_s=max_eval_time_delta_s,
            window_s=window_s,
            top_n=top_n,
        ),
    }


def _source_mask(frame: pd.DataFrame, source: str) -> pd.Series:
    if "source" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["source"].astype(str) == source


def _top_residuals(frame: pd.DataFrame, *, top_n: int) -> list[dict[str, Any]]:
    if frame.empty or "residual_norm_m" not in frame.columns:
        return []
    work = frame.copy()
    work["_residual_norm_m"] = pd.to_numeric(work["residual_norm_m"], errors="coerce")
    work = work.loc[np.isfinite(work["_residual_norm_m"])]
    if work.empty:
        return []
    work = work.sort_values("_residual_norm_m", ascending=False).head(top_n)
    columns = [
        "time_s",
        "source",
        "track_id",
        "measurement_dim",
        "accepted",
        "update_action",
        "residual_norm_m",
        "nis",
        "gate_threshold",
        "covariance_scale",
        "inflation_alpha",
        "association_nis",
        "association_score",
        "hypothesis_count",
    ]
    return [_row_payload(row, columns) for _, row in work.iterrows()]


def _track_switch_summary(frame: pd.DataFrame, *, top_n: int) -> dict[str, Any]:
    if frame.empty or "track_id" not in frame.columns:
        return _empty_track_switch_summary()

    work = frame.copy()
    if "time_s" in work.columns:
        work["_time_s"] = pd.to_numeric(work["time_s"], errors="coerce")
        work = work.sort_values("_time_s")
    track_ids = pd.to_numeric(work["track_id"], errors="coerce")
    finite = track_ids[np.isfinite(track_ids)].astype(int)
    if finite.empty:
        return _empty_track_switch_summary()

    events: list[dict[str, Any]] = []
    transitions: Counter[tuple[int, int]] = Counter()
    previous: int | None = None
    for index, current in finite.items():
        current_id = int(current)
        if previous is not None and current_id != previous:
            transitions[(previous, current_id)] += 1
            event: dict[str, Any] = {"from_track_id": previous, "to_track_id": current_id}
            if "time_s" in work.columns:
                event["time_s"] = _json_value(work.loc[index, "time_s"])
            events.append(event)
        previous = current_id

    return {
        "count": int(sum(transitions.values())),
        "updates_with_track_id": int(len(finite)),
        "unique_track_ids": int(finite.nunique()),
        "first_track_id": int(finite.iloc[0]),
        "last_track_id": int(finite.iloc[-1]),
        "top_transitions": [
            {"from_track_id": int(src), "to_track_id": int(dst), "count": int(count)}
            for (src, dst), count in transitions.most_common(top_n)
        ],
        "events": events[:top_n],
    }


def _empty_track_switch_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "updates_with_track_id": 0,
        "unique_track_ids": 0,
        "first_track_id": None,
        "last_track_id": None,
        "top_transitions": [],
        "events": [],
    }


def _covariance_inflation_summary(frame: pd.DataFrame, *, top_n: int) -> dict[str, Any]:
    if frame.empty or "covariance_scale" not in frame.columns:
        return {
            "count": 0,
            "by_source": {},
            "mean_scale": None,
            "p95_scale": None,
            "max_scale": None,
            "top_scaled_updates": [],
        }

    work = frame.copy()
    work["_covariance_scale"] = pd.to_numeric(work["covariance_scale"], errors="coerce")
    inflated = work.loc[np.isfinite(work["_covariance_scale"]) & (work["_covariance_scale"] > 1.0)]
    if inflated.empty:
        return {
            "count": 0,
            "by_source": {},
            "mean_scale": None,
            "p95_scale": None,
            "max_scale": None,
            "top_scaled_updates": [],
        }

    source_counts: dict[str, int] = {}
    if "source" in inflated.columns:
        source_counts = {
            str(source): int(count)
            for source, count in inflated["source"].astype(str).value_counts().sort_index().items()
        }
    scale = inflated["_covariance_scale"].to_numpy(dtype=float)
    top = inflated.sort_values("_covariance_scale", ascending=False).head(top_n)
    columns = [
        "time_s",
        "source",
        "track_id",
        "measurement_dim",
        "update_action",
        "covariance_scale",
        "residual_norm_m",
        "nis",
        "gate_threshold",
    ]
    return {
        "count": int(len(inflated)),
        "by_source": source_counts,
        "mean_scale": float(np.mean(scale)),
        "p95_scale": float(np.percentile(scale, 95)),
        "max_scale": float(np.max(scale)),
        "top_scaled_updates": [_row_payload(row, columns) for _, row in top.iterrows()],
    }


def _worst_time_windows(
    *,
    estimate_frame: pd.DataFrame,
    truth: pd.DataFrame,
    max_eval_time_delta_s: float | None,
    window_s: float,
    top_n: int,
) -> list[dict[str, Any]]:
    errors = _position_error_frame(
        estimate_frame=estimate_frame,
        truth=truth,
        max_eval_time_delta_s=max_eval_time_delta_s,
    )
    if errors.empty:
        return []

    errors["window_start_s"] = np.floor(errors["time_s"].to_numpy(dtype=float) / window_s) * window_s
    rows: list[dict[str, Any]] = []
    for window_start, group in errors.groupby("window_start_s", sort=True):
        error_3d = group["error_3d_m"].to_numpy(dtype=float)
        residual = group["residual_norm_m"].dropna().to_numpy(dtype=float)
        covariance_scale = group["covariance_scale"].dropna().to_numpy(dtype=float)
        track_switches = _track_switch_summary(
            group.loc[_source_mask(group, "radar")],
            top_n=1,
        )["count"]
        rows.append(
            {
                "time_start_s": float(window_start),
                "time_end_s": float(window_start + window_s),
                "count": int(len(group)),
                "rmse_3d_m": float(np.sqrt(np.mean(error_3d**2))),
                "mae_3d_m": float(np.mean(np.abs(error_3d))),
                "p95_3d_m": float(np.percentile(error_3d, 95)),
                "max_3d_m": float(np.max(error_3d)),
                "mean_residual_norm_m": None
                if residual.size == 0
                else float(np.mean(residual)),
                "covariance_inflation_count": int(np.sum(covariance_scale > 1.0)),
                "track_switch_count": int(track_switches),
            }
        )
    rows.sort(key=lambda item: item["rmse_3d_m"], reverse=True)
    return rows[:top_n]


def _position_error_frame(
    *,
    estimate_frame: pd.DataFrame,
    truth: pd.DataFrame,
    max_eval_time_delta_s: float | None,
) -> pd.DataFrame:
    required_estimate = {"time_s", "east_m", "north_m", "up_m"}
    required_truth = {"time_s", "east_m", "north_m", "up_m"}
    if estimate_frame.empty or truth.empty:
        return pd.DataFrame()
    if not required_estimate.issubset(estimate_frame.columns):
        return pd.DataFrame()
    if not required_truth.issubset(truth.columns):
        return pd.DataFrame()

    estimate_times = estimate_frame["time_s"].to_numpy(dtype=float)
    estimate_positions = estimate_frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    truth_indices = nearest_time_indices(truth_times, estimate_times)
    time_delta = np.abs(truth_times[truth_indices] - estimate_times)
    deltas = estimate_positions - truth_positions[truth_indices]
    error_2d = np.linalg.norm(deltas[:, :2], axis=1)
    error_3d = np.linalg.norm(deltas, axis=1)
    finite = (
        np.isfinite(estimate_times)
        & np.isfinite(time_delta)
        & np.isfinite(error_2d)
        & np.isfinite(error_3d)
    )
    if max_eval_time_delta_s is not None:
        finite &= time_delta <= float(max_eval_time_delta_s)

    out = estimate_frame.loc[finite].copy()
    out["truth_time_delta_s"] = time_delta[finite]
    out["error_2d_m"] = error_2d[finite]
    out["error_3d_m"] = error_3d[finite]
    return out


def _row_payload(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for column in columns:
        if column in row.index:
            payload[column] = _json_value(row[column])
    return payload


def _json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
