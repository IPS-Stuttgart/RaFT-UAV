"""Radar row association for the asynchronous CV baseline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    gate_threshold_from_probability,
    measurement_matrix,
)

RADAR_ASSOCIATION_MODES = (
    "oracle-nearest-truth",
    "prediction-nis",
    "track-continuity",
    "geometry-score",
    "pda-mixture",
)


def run_async_cv_baseline_with_radar_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    association: str,
    truth: pd.DataFrame | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    track_switch_nis_ratio: float = 0.5,
    candidate_catprob_threshold: float | None = 0.5,
    geometry_velocity_std_mps: float = 12.0,
    geometry_velocity_weight: float = 0.25,
    geometry_switch_penalty: float = 4.0,
    geometry_catprob_weight: float = 2.0,
    pda_nis_temperature: float = 1.0,
    pda_catprob_exponent: float = 1.0,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion while selecting at most one radar row per radar frame.

    ``oracle-nearest-truth`` uses ground truth and is only a diagnostic upper
    bound. ``prediction-nis`` picks the radar candidate with the lowest
    normalized innovation squared against the current predicted state.
    ``track-continuity`` prefers the current Fortem track ID and switches only
    when another candidate has a substantially lower NIS. ``geometry-score``
    is an online score that augments NIS with radar velocity consistency,
    track switching, and UAV class-probability penalties. ``pda-mixture``
    keeps a single Kalman update but forms it from a probability-weighted
    candidate mixture and adds candidate spread to the radar covariance.
    """

    if association not in RADAR_ASSOCIATION_MODES:
        raise ValueError(f"unknown radar association mode {association!r}")
    if association == "oracle-nearest-truth" and truth is None:
        raise ValueError("oracle-nearest-truth association requires normalized truth")
    if track_switch_nis_ratio <= 0.0:
        raise ValueError("track_switch_nis_ratio must be positive")
    if geometry_velocity_std_mps <= 0.0:
        raise ValueError("geometry_velocity_std_mps must be positive")
    for name, value in {
        "geometry_velocity_weight": geometry_velocity_weight,
        "geometry_switch_penalty": geometry_switch_penalty,
        "geometry_catprob_weight": geometry_catprob_weight,
    }.items():
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if pda_nis_temperature <= 0.0:
        raise ValueError("pda_nis_temperature must be positive")
    if pda_catprob_exponent < 0.0:
        raise ValueError("pda_catprob_exponent must be nonnegative")

    covariance = np.diag([float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2])
    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _empty_selected_radar(radar)

    initial_measurement = _initial_measurement(
        events[0],
        association=association,
        covariance=covariance,
        truth=truth,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )
    if initial_measurement is None:
        return [], _empty_selected_radar(radar)

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    current_track_id: int | None = None

    for event in events:
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(
                measurement,
                gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=gate_probabilities_by_source,
                    gate_thresholds_by_source=gate_thresholds_by_source,
                ),
                robust_update=_robust_update_for_measurement(
                    measurement,
                    robust_update_by_source=robust_update_by_source,
                ),
                inflation_alpha=_inflation_alpha_for_measurement(
                    measurement,
                    inflation_alpha_by_source=inflation_alpha_by_source,
                ),
            )
            records.append(_record(measurement, tracker, diagnostics))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        time_s = float(event["time_s"])
        tracker.predict_to(time_s)
        selected = _select_radar_candidate(
            candidates,
            association=association,
            tracker=tracker,
            covariance=covariance,
            truth=truth,
            current_track_id=current_track_id,
            track_switch_nis_ratio=track_switch_nis_ratio,
            candidate_catprob_threshold=candidate_catprob_threshold,
            geometry_velocity_std_mps=geometry_velocity_std_mps,
            geometry_velocity_weight=geometry_velocity_weight,
            geometry_switch_penalty=geometry_switch_penalty,
            geometry_catprob_weight=geometry_catprob_weight,
            pda_nis_temperature=pda_nis_temperature,
            pda_catprob_exponent=pda_catprob_exponent,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        if selected is None:
            continue

        measurement = _radar_row_to_measurement(selected, covariance)
        diagnostics = tracker.update(
            measurement,
            gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=gate_probabilities_by_source,
                gate_thresholds_by_source=gate_thresholds_by_source,
            ),
            robust_update=_robust_update_for_measurement(
                measurement,
                robust_update_by_source=robust_update_by_source,
            ),
            inflation_alpha=_inflation_alpha_for_measurement(
                measurement,
                inflation_alpha_by_source=inflation_alpha_by_source,
            ),
        )
        if diagnostics.accepted:
            current_track_id = _optional_track_id(selected)
        selected_rows.append(selected)
        records.append(
            _record(
                measurement,
                tracker,
                diagnostics,
                track_id=_optional_track_id(selected),
                association_nis=_optional_float(selected.get("association_nis")),
                association_score=_optional_float(selected.get("association_score")),
                association_mode=association,
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)


def _events(
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        {"time_s": measurement.time_s, "priority": 0, "kind": "rf", "measurement": measurement}
        for measurement in rf_measurements
    ]
    for group in _radar_frame_groups(radar):
        events.append(
            {
                "time_s": float(group["time_s"].median()),
                "priority": 1,
                "kind": "radar",
                "candidates": group,
            }
        )
    return sorted(events, key=lambda item: (float(item["time_s"]), int(item["priority"])))


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _initial_measurement(
    event: dict[str, object],
    *,
    association: str,
    covariance: np.ndarray,
    truth: pd.DataFrame | None,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> TrackingMeasurement | None:
    if event["kind"] == "rf":
        measurement = event["measurement"]
        assert isinstance(measurement, TrackingMeasurement)
        return measurement
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if association == "oracle-nearest-truth":
        selected = _oracle_nearest_truth(
            candidates,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
    else:
        selected = _highest_catprob(candidates)
    if selected is None:
        return None
    return _radar_row_to_measurement(selected, covariance)


def _select_radar_candidate(
    candidates: pd.DataFrame,
    *,
    association: str,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
    truth: pd.DataFrame | None,
    current_track_id: int | None,
    track_switch_nis_ratio: float,
    candidate_catprob_threshold: float | None,
    geometry_velocity_std_mps: float,
    geometry_velocity_weight: float,
    geometry_switch_penalty: float,
    geometry_catprob_weight: float,
    pda_nis_temperature: float,
    pda_catprob_exponent: float,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.Series | None:
    if candidates.empty:
        return None
    if association == "oracle-nearest-truth":
        return _oracle_nearest_truth(
            candidates,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )

    candidates = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    if candidates.empty:
        return None
    scored = _nis_scored_candidates(candidates, tracker, covariance)
    if scored.empty:
        return None
    if association == "geometry-score":
        geometry_scored = _geometry_scored_candidates(
            scored,
            tracker=tracker,
            current_track_id=current_track_id,
            velocity_std_mps=geometry_velocity_std_mps,
            velocity_weight=geometry_velocity_weight,
            switch_penalty=geometry_switch_penalty,
            catprob_weight=geometry_catprob_weight,
        )
        best = geometry_scored.loc[geometry_scored["association_score"].idxmin()].copy()
        best["association_action"] = "geometry_score"
        return best
    if association == "pda-mixture":
        return _pda_mixture_candidate(
            scored,
            base_covariance=covariance,
            nis_temperature=pda_nis_temperature,
            catprob_exponent=pda_catprob_exponent,
        )
    best = scored.loc[scored["association_nis"].idxmin()].copy()
    if association == "prediction-nis" or current_track_id is None:
        return best
    if association != "track-continuity":
        raise ValueError(f"unknown radar association mode {association!r}")

    current = scored.loc[scored["track_id"] == current_track_id]
    if current.empty:
        return best
    current_best = current.loc[current["association_nis"].idxmin()].copy()
    if int(best["track_id"]) == current_track_id:
        return best
    if float(best["association_nis"]) < float(current_best["association_nis"]) * float(
        track_switch_nis_ratio
    ):
        return best
    current_best["association_action"] = "kept_track"
    return current_best


def _oracle_nearest_truth(
    candidates: pd.DataFrame,
    *,
    truth: pd.DataFrame | None,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.Series | None:
    if truth is None or truth.empty:
        return None
    truth_xyz = _nearest_truth_position(
        truth,
        time_s=float(candidates["time_s"].median()),
        max_delta_s=float(truth_time_gate_s),
    )
    if truth_xyz is None:
        return None
    candidate_xyz = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
    best_position = int(np.argmin(errors))
    if float(errors[best_position]) > float(truth_gate_m):
        return None
    selected = candidates.iloc[best_position].copy()
    selected["association_nis"] = float(errors[best_position])
    selected["association_truth_error_m"] = float(errors[best_position])
    selected["association_mode"] = "oracle-nearest-truth"
    selected["association_candidate_rows"] = int(len(candidates))
    return selected


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    candidate_catprob_threshold: float | None,
) -> pd.DataFrame:
    if candidate_catprob_threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    return candidates.loc[catprob >= float(candidate_catprob_threshold)].copy()


def _highest_catprob(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    selected: pd.Series
    if "cat_prob_uav" in candidates.columns:
        scores = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(-np.inf)
        selected = candidates.loc[scores.idxmax()].copy()
    else:
        selected = candidates.iloc[0].copy()
    selected["association_mode"] = "bootstrap-catprob"
    selected["association_candidate_rows"] = int(len(candidates))
    return selected


def _nis_scored_candidates(
    candidates: pd.DataFrame,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.iloc[0:0].copy()
    observation = measurement_matrix(3)
    state_position = observation @ tracker.state
    innovation_covariance = observation @ tracker.covariance_matrix @ observation.T + covariance
    try:
        precision = np.linalg.inv(innovation_covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(innovation_covariance)
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = vectors - state_position
    scored = candidates.copy()
    scored["association_nis"] = np.einsum("ij,jk,ik->i", residuals, precision, residuals)
    scored["association_candidate_rows"] = int(len(candidates))
    return scored


def _geometry_scored_candidates(
    candidates: pd.DataFrame,
    *,
    tracker: AsyncConstantVelocityKalmanTracker,
    current_track_id: int | None,
    velocity_std_mps: float,
    velocity_weight: float,
    switch_penalty: float,
    catprob_weight: float,
) -> pd.DataFrame:
    scored = candidates.copy()
    velocity_nis = _candidate_velocity_nis(scored, tracker.state[3:6], velocity_std_mps)
    scored["association_velocity_nis"] = velocity_nis
    scored["association_velocity_penalty"] = float(velocity_weight) * velocity_nis
    scored["association_switch_penalty"] = _track_switch_penalty(
        scored,
        current_track_id=current_track_id,
        switch_penalty=switch_penalty,
    )
    scored["association_catprob_penalty"] = _catprob_penalty(scored, catprob_weight)
    scored["association_score"] = (
        scored["association_nis"]
        + scored["association_velocity_penalty"]
        + scored["association_switch_penalty"]
        + scored["association_catprob_penalty"]
    )
    return scored


def _candidate_velocity_nis(
    candidates: pd.DataFrame,
    predicted_velocity_enu_mps: np.ndarray,
    velocity_std_mps: float,
) -> np.ndarray:
    required = {"velocity_east_mps", "velocity_north_mps", "velocity_down_mps"}
    if not required.issubset(candidates.columns):
        return np.zeros(len(candidates), dtype=float)
    velocities = np.column_stack(
        [
            pd.to_numeric(candidates["velocity_east_mps"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(candidates["velocity_north_mps"], errors="coerce").to_numpy(dtype=float),
            -pd.to_numeric(candidates["velocity_down_mps"], errors="coerce").to_numpy(dtype=float),
        ]
    )
    finite = np.isfinite(velocities).all(axis=1)
    residuals = velocities - np.asarray(predicted_velocity_enu_mps, dtype=float).reshape(3)
    velocity_nis = np.sum((residuals / float(velocity_std_mps)) ** 2, axis=1)
    return np.where(finite, velocity_nis, 0.0)


def _track_switch_penalty(
    candidates: pd.DataFrame,
    *,
    current_track_id: int | None,
    switch_penalty: float,
) -> np.ndarray:
    if current_track_id is None or "track_id" not in candidates.columns:
        return np.zeros(len(candidates), dtype=float)
    track_ids = pd.to_numeric(candidates["track_id"], errors="coerce").to_numpy(dtype=float)
    switches = np.zeros(len(candidates), dtype=bool)
    finite = np.isfinite(track_ids)
    switches[finite] = track_ids[finite].astype(int) != int(current_track_id)
    return np.where(switches, float(switch_penalty), 0.0)


def _catprob_penalty(candidates: pd.DataFrame, catprob_weight: float) -> np.ndarray:
    if "cat_prob_uav" not in candidates.columns:
        return np.zeros(len(candidates), dtype=float)
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(1.0)
    catprob = np.clip(catprob.to_numpy(dtype=float), 0.0, 1.0)
    return float(catprob_weight) * (1.0 - catprob) ** 2


def _pda_mixture_candidate(
    candidates: pd.DataFrame,
    *,
    base_covariance: np.ndarray,
    nis_temperature: float,
    catprob_exponent: float,
) -> pd.Series:
    weights = _pda_weights(
        candidates,
        nis_temperature=nis_temperature,
        catprob_exponent=catprob_exponent,
    )
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    mean = weights @ vectors
    residuals = vectors - mean
    spread = residuals.T @ (residuals * weights[:, None])
    covariance = np.asarray(base_covariance, dtype=float) + spread

    best_index = int(np.argmax(weights))
    selected = candidates.iloc[best_index].copy()
    selected["east_m"] = float(mean[0])
    selected["north_m"] = float(mean[1])
    selected["up_m"] = float(mean[2])
    selected["association_mode"] = "pda-mixture"
    selected["association_action"] = "pda_mixture"
    selected["association_candidate_rows"] = int(len(candidates))
    selected["association_nis"] = float(
        weights @ candidates["association_nis"].to_numpy(dtype=float)
    )
    selected["association_score"] = float(-np.log(float(np.max(weights))))
    selected["association_weight_max"] = float(np.max(weights))
    selected["association_weight_entropy"] = _weight_entropy(weights)
    selected["association_effective_candidates"] = float(1.0 / np.sum(weights**2))
    selected["association_best_track_id"] = _optional_track_id(selected)
    selected["association_position_spread_trace_m2"] = float(np.trace(spread))
    selected["association_cov_ee"] = float(covariance[0, 0])
    selected["association_cov_nn"] = float(covariance[1, 1])
    selected["association_cov_uu"] = float(covariance[2, 2])
    selected["association_cov_en"] = float(covariance[0, 1])
    selected["association_cov_eu"] = float(covariance[0, 2])
    selected["association_cov_nu"] = float(covariance[1, 2])
    return selected


def _pda_weights(
    candidates: pd.DataFrame,
    *,
    nis_temperature: float,
    catprob_exponent: float,
) -> np.ndarray:
    nis = pd.to_numeric(candidates["association_nis"], errors="coerce").to_numpy(dtype=float)
    log_weights = -0.5 * nis / float(nis_temperature)
    if "cat_prob_uav" in candidates.columns and catprob_exponent > 0.0:
        catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(1e-3)
        catprob = np.clip(catprob.to_numpy(dtype=float), 1e-3, 1.0)
        log_weights = log_weights + float(catprob_exponent) * np.log(catprob)
    log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    maximum = float(np.max(log_weights))
    if not np.isfinite(maximum):
        return np.full(len(candidates), 1.0 / len(candidates))
    weights = np.exp(log_weights - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(candidates), 1.0 / len(candidates))
    return weights / total


def _weight_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-300, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _radar_row_to_measurement(row: pd.Series, covariance: np.ndarray) -> TrackingMeasurement:
    row_covariance = _row_covariance(row)
    return TrackingMeasurement(
        time_s=float(row["time_s"]),
        vector=np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])]),
        covariance=covariance if row_covariance is None else row_covariance,
        source="radar",
    )


def _row_covariance(row: pd.Series) -> np.ndarray | None:
    columns = [
        "association_cov_ee",
        "association_cov_nn",
        "association_cov_uu",
        "association_cov_en",
        "association_cov_eu",
        "association_cov_nu",
    ]
    if not all(column in row for column in columns):
        return None
    values = [float(row[column]) for column in columns]
    if not np.isfinite(values).all():
        return None
    ee, nn, uu, en, eu, nu = values
    return np.array(
        [
            [ee, en, eu],
            [en, nn, nu],
            [eu, nu, uu],
        ],
        dtype=float,
    )


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_delta_s: float,
) -> np.ndarray | None:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    if truth_times.size == 0:
        return None
    insertion = np.searchsorted(truth_times, float(time_s))
    right = int(np.clip(insertion, 0, truth_times.size - 1))
    left = int(np.clip(insertion - 1, 0, truth_times.size - 1))
    nearest = right if abs(truth_times[right] - time_s) < abs(truth_times[left] - time_s) else left
    if abs(truth_times[nearest] - time_s) > max_delta_s:
        return None
    return truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[nearest]


def _gate_threshold_for_measurement(
    measurement: TrackingMeasurement,
    *,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
) -> float | None:
    if gate_thresholds_by_source and measurement.source in gate_thresholds_by_source:
        threshold = gate_thresholds_by_source[measurement.source]
        return None if threshold is None else float(threshold)
    if gate_probabilities_by_source and measurement.source in gate_probabilities_by_source:
        return gate_threshold_from_probability(
            gate_probabilities_by_source[measurement.source],
            measurement.vector.size,
        )
    return None


def _robust_update_for_measurement(
    measurement: TrackingMeasurement,
    *,
    robust_update_by_source: Mapping[str, str | None] | None,
) -> str | None:
    if robust_update_by_source and measurement.source in robust_update_by_source:
        return robust_update_by_source[measurement.source]
    return None


def _inflation_alpha_for_measurement(
    measurement: TrackingMeasurement,
    *,
    inflation_alpha_by_source: Mapping[str, float] | None,
) -> float:
    if inflation_alpha_by_source and measurement.source in inflation_alpha_by_source:
        return float(inflation_alpha_by_source[measurement.source])
    return 1.0


def _record(
    measurement: TrackingMeasurement,
    tracker: AsyncConstantVelocityKalmanTracker,
    diagnostics: Any,
    *,
    track_id: int | None = None,
    association_nis: float | None = None,
    association_score: float | None = None,
    association_mode: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "time_s": measurement.time_s,
        "source": measurement.source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        **diagnostics.to_record(),
    }
    if track_id is not None:
        record["track_id"] = track_id
    if association_nis is not None:
        record["association_nis"] = association_nis
    if association_score is not None:
        record["association_score"] = association_score
    if association_mode is not None:
        record["association_mode"] = association_mode
    return record


def _selected_rows_frame(radar: pd.DataFrame, rows: list[pd.Series]) -> pd.DataFrame:
    if not rows:
        return _empty_selected_radar(radar)
    selected = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _empty_selected_radar(radar: pd.DataFrame) -> pd.DataFrame:
    selected = radar.iloc[0:0].copy()
    for column in (
        "association_mode",
        "association_nis",
        "association_score",
        "association_candidate_rows",
        "association_effective_candidates",
        "association_weight_max",
        "association_position_spread_trace_m2",
    ):
        selected[column] = []
    return selected


def _optional_track_id(row: pd.Series) -> int | None:
    value = row.get("track_id")
    if value is None or not np.isfinite(float(value)):
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
