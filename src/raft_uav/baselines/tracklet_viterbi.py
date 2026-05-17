"""Radar tracklet-level Viterbi association.

This module performs association before filtering by selecting one globally
consistent Fortem radar row per radar frame.  It is intentionally independent
of PyRecEst so it can be tested as a deterministic scoring primitive.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackletViterbiConfig:
    """Cost weights for radar tracklet Viterbi association."""

    max_candidates_per_frame: int = 12
    transition_std_m: float = 60.0
    velocity_std_mps: float = 15.0
    switch_penalty: float = 9.0
    same_track_reward: float = 1.5
    catprob_weight: float = 6.0
    track_length_reward: float = 0.35
    rf_support_weight: float = 0.4
    rf_support_std_m: float = 250.0
    rf_time_gate_s: float = 2.0
    max_speed_mps: float = 55.0
    speed_penalty_weight: float = 4.0


def select_tracklet_viterbi_path(
    radar: pd.DataFrame,
    *,
    rf_measurements: Iterable[Any] = (),
    candidate_catprob_threshold: float | None = 0.5,
    config: TrackletViterbiConfig | None = None,
) -> pd.DataFrame:
    """Select a dynamically consistent single-UAV radar path.

    The returned frame contains one selected row per radar frame.  Rows are
    annotated with Viterbi diagnostics and can be fed into the existing CV
    Kalman fusion code as if they were ordinary radar measurements.
    """

    cfg = config or TrackletViterbiConfig()
    _validate_config(cfg)
    if radar.empty:
        return _empty_selection(radar)

    frame_groups = _radar_frame_groups(radar)
    track_lengths = _track_lengths(radar)
    rf_times, rf_xy = _rf_arrays(rf_measurements)
    layers = [
        _candidate_layer(
            frame,
            track_lengths=track_lengths,
            rf_times=rf_times,
            rf_xy=rf_xy,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
        )
        for frame in frame_groups
    ]
    layers = [layer for layer in layers if layer]
    if not layers:
        return _empty_selection(radar)

    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    transition_costs: list[np.ndarray] = []
    costs.append(np.array([candidate["local_cost"] for candidate in layers[0]], dtype=float))
    backpointers.append(np.full(len(layers[0]), -1, dtype=int))
    transition_costs.append(np.zeros(len(layers[0]), dtype=float))

    for layer_index in range(1, len(layers)):
        previous_layer = layers[layer_index - 1]
        current_layer = layers[layer_index]
        previous_costs = costs[-1]
        current_costs = np.empty(len(current_layer), dtype=float)
        current_backpointers = np.empty(len(current_layer), dtype=int)
        current_transition_costs = np.empty(len(current_layer), dtype=float)
        for candidate_index, candidate in enumerate(current_layer):
            transitions = np.array(
                [
                    _transition_cost(previous, candidate, cfg)
                    for previous in previous_layer
                ],
                dtype=float,
            )
            totals = previous_costs + transitions
            best_previous = int(np.argmin(totals))
            current_costs[candidate_index] = (
                float(totals[best_previous]) + float(candidate["local_cost"])
            )
            current_backpointers[candidate_index] = best_previous
            current_transition_costs[candidate_index] = float(transitions[best_previous])
        costs.append(current_costs)
        backpointers.append(current_backpointers)
        transition_costs.append(current_transition_costs)

    selected_indices = _backtrace(costs, backpointers)
    rows: list[pd.Series] = []
    total_cost = float(costs[-1][selected_indices[-1]])
    for layer_index, candidate_index in enumerate(selected_indices):
        candidate = layers[layer_index][candidate_index]
        row = candidate["row"].copy()
        row["association_mode"] = "tracklet-viterbi"
        row["association_action"] = "viterbi_path"
        row["association_score"] = total_cost
        row["association_viterbi_total_cost"] = total_cost
        row["association_viterbi_local_cost"] = float(candidate["local_cost"])
        row["association_viterbi_transition_cost"] = float(
            transition_costs[layer_index][candidate_index]
        )
        row["association_candidate_rows"] = int(candidate["candidate_rows"])
        row["association_catprob_threshold"] = (
            np.nan
            if candidate_catprob_threshold is None
            else float(candidate_catprob_threshold)
        )
        row["association_catprob_fallback"] = bool(candidate["catprob_fallback"])
        row["association_track_length"] = int(candidate["track_length"])
        row["association_rf_support_cost"] = float(candidate["rf_support_cost"])
        rows.append(row)

    return _selected_rows_frame(pd.DataFrame(rows))


def _validate_config(config: TrackletViterbiConfig) -> None:
    if config.max_candidates_per_frame < 1:
        raise ValueError("tracklet_viterbi_max_candidates must be positive")
    for name in (
        "transition_std_m",
        "velocity_std_mps",
        "rf_support_std_m",
        "rf_time_gate_s",
        "max_speed_mps",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"tracklet_viterbi_{name} must be positive")
    for name in (
        "switch_penalty",
        "same_track_reward",
        "catprob_weight",
        "track_length_reward",
        "rf_support_weight",
        "speed_penalty_weight",
    ):
        if getattr(config, name) < 0.0:
            raise ValueError(f"tracklet_viterbi_{name} must be nonnegative")


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _track_lengths(radar: pd.DataFrame) -> dict[int, int]:
    if "track_id" not in radar.columns:
        return {}
    counts: dict[int, int] = {}
    for value in pd.to_numeric(radar["track_id"], errors="coerce").dropna().to_numpy(dtype=float):
        counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _candidate_layer(
    frame: pd.DataFrame,
    *,
    track_lengths: dict[int, int],
    rf_times: np.ndarray,
    rf_xy: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiConfig,
) -> list[dict[str, Any]]:
    pool, catprob_fallback = _catprob_candidate_pool(frame, candidate_catprob_threshold)
    candidates: list[dict[str, Any]] = []
    for _, row in pool.iterrows():
        track_id = _optional_track_id(row)
        track_length = 1 if track_id is None else track_lengths.get(track_id, 1)
        catprob_cost = _catprob_cost(row, config.catprob_weight)
        track_length_reward = float(config.track_length_reward) * np.log1p(float(track_length))
        rf_support_cost = _rf_support_cost(row, rf_times, rf_xy, config)
        local_cost = catprob_cost + rf_support_cost - track_length_reward
        candidates.append(
            {
                "row": row.copy(),
                "local_cost": float(local_cost),
                "rf_support_cost": float(rf_support_cost),
                "track_length": int(track_length),
                "candidate_rows": int(len(frame)),
                "catprob_fallback": bool(catprob_fallback),
            }
        )
    candidates.sort(key=lambda item: float(item["local_cost"]))
    return candidates[: int(config.max_candidates_per_frame)]


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    threshold: float | None,
) -> tuple[pd.DataFrame, bool]:
    if threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates.copy(), False
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    keep = catprob >= float(threshold)
    if keep.any():
        return candidates.loc[keep].copy(), False
    return candidates.copy(), True


def _catprob_cost(row: pd.Series, weight: float) -> float:
    if "cat_prob_uav" not in row:
        return 0.0
    probability = _finite_float(row.get("cat_prob_uav"), default=0.5)
    probability = float(np.clip(probability, 1.0e-3, 1.0))
    return float(weight) * float(-np.log(probability))


def _rf_arrays(rf_measurements: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[list[float]] = []
    for measurement in rf_measurements:
        try:
            vector = np.asarray(measurement.vector, dtype=float).reshape(-1)
            time_s = float(measurement.time_s)
        except (AttributeError, TypeError, ValueError):
            continue
        if vector.size < 2 or not np.isfinite(vector[:2]).all() or not np.isfinite(time_s):
            continue
        times.append(time_s)
        positions.append([float(vector[0]), float(vector[1])])
    if not times:
        return np.empty(0, dtype=float), np.empty((0, 2), dtype=float)
    order = np.argsort(np.asarray(times, dtype=float))
    return np.asarray(times, dtype=float)[order], np.asarray(positions, dtype=float)[order]


def _rf_support_cost(
    row: pd.Series,
    rf_times: np.ndarray,
    rf_xy: np.ndarray,
    config: TrackletViterbiConfig,
) -> float:
    if rf_times.size == 0 or config.rf_support_weight == 0.0:
        return 0.0
    time_s = _finite_float(row.get("time_s"), default=np.nan)
    if not np.isfinite(time_s):
        return 0.0
    insertion = int(np.searchsorted(rf_times, time_s))
    nearest_indices = [
        index
        for index in (insertion - 1, insertion)
        if 0 <= index < rf_times.size
    ]
    if not nearest_indices:
        return 0.0
    nearest = min(nearest_indices, key=lambda index: abs(float(rf_times[index]) - time_s))
    if abs(float(rf_times[nearest]) - time_s) > float(config.rf_time_gate_s):
        return 0.0
    position = _position(row)
    residual_2d = float(np.linalg.norm(position[:2] - rf_xy[nearest]))
    normalized = (residual_2d / float(config.rf_support_std_m)) ** 2
    return float(config.rf_support_weight) * min(normalized, 9.0)


def _transition_cost(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: TrackletViterbiConfig,
) -> float:
    previous_row = previous["row"]
    current_row = current["row"]
    previous_position = _position(previous_row)
    current_position = _position(current_row)
    dt_s = max(_time(current_row) - _time(previous_row), 1.0e-6)
    previous_velocity = _velocity(previous_row)
    current_velocity = _velocity(current_row)

    if previous_velocity is None:
        expected_position = previous_position
        position_scale = float(config.transition_std_m) + dt_s * float(config.max_speed_mps)
    else:
        expected_position = previous_position + previous_velocity * dt_s
        position_scale = float(config.transition_std_m) + dt_s * float(config.velocity_std_mps)
    position_residual = float(np.linalg.norm(current_position - expected_position))
    motion_cost = (position_residual / position_scale) ** 2

    implied_speed = float(np.linalg.norm(current_position - previous_position) / dt_s)
    speed_cost = 0.0
    if implied_speed > float(config.max_speed_mps):
        overspeed = (implied_speed - float(config.max_speed_mps)) / float(config.velocity_std_mps)
        speed_cost = float(config.speed_penalty_weight) * overspeed**2

    velocity_cost = 0.0
    if previous_velocity is not None and current_velocity is not None:
        velocity_cost = float(
            np.sum(((current_velocity - previous_velocity) / float(config.velocity_std_mps)) ** 2)
        )

    continuity_cost = _track_continuity_cost(previous_row, current_row, config)
    return float(motion_cost + speed_cost + 0.25 * velocity_cost + continuity_cost)


def _track_continuity_cost(
    previous: pd.Series,
    current: pd.Series,
    config: TrackletViterbiConfig,
) -> float:
    previous_id = _optional_track_id(previous)
    current_id = _optional_track_id(current)
    if previous_id is None or current_id is None:
        return 0.0
    if previous_id == current_id:
        return -float(config.same_track_reward)
    return float(config.switch_penalty)


def _position(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            _finite_float(row.get("east_m"), default=0.0),
            _finite_float(row.get("north_m"), default=0.0),
            _finite_float(row.get("up_m"), default=0.0),
        ],
        dtype=float,
    )


def _velocity(row: pd.Series) -> np.ndarray | None:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in row for column in required):
        return None
    values = np.array(
        [
            _finite_float(row.get("velocity_east_mps"), default=np.nan),
            _finite_float(row.get("velocity_north_mps"), default=np.nan),
            -_finite_float(row.get("velocity_down_mps"), default=np.nan),
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        return None
    return values


def _time(row: pd.Series) -> float:
    return _finite_float(row.get("time_s"), default=0.0)


def _backtrace(costs: list[np.ndarray], backpointers: list[np.ndarray]) -> list[int]:
    selected = [int(np.argmin(costs[-1]))]
    for layer_index in range(len(costs) - 1, 0, -1):
        selected.append(int(backpointers[layer_index][selected[-1]]))
    return list(reversed(selected))


def _selected_rows_frame(selected: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _empty_selection(radar: pd.DataFrame) -> pd.DataFrame:
    selected = radar.iloc[0:0].copy()
    for column in (
        "association_mode",
        "association_action",
        "association_score",
        "association_viterbi_total_cost",
        "association_viterbi_local_cost",
        "association_viterbi_transition_cost",
        "association_candidate_rows",
        "association_catprob_threshold",
        "association_catprob_fallback",
        "association_track_length",
        "association_rf_support_cost",
    ):
        selected[column] = []
    return selected


def _optional_track_id(row: pd.Series) -> int | None:
    if "track_id" not in row:
        return None
    value = _finite_float(row.get("track_id"), default=np.nan)
    if not np.isfinite(value):
        return None
    return int(value)


def _finite_float(value: object, *, default: float) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result
