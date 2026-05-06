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
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion while selecting at most one radar row per radar frame.

    ``oracle-nearest-truth`` uses ground truth and is only a diagnostic upper
    bound. ``prediction-nis`` picks the radar candidate with the lowest
    normalized innovation squared against the current predicted state.
    ``track-continuity`` prefers the current Fortem track ID and switches only
    when another candidate has a substantially lower NIS.
    """

    if association not in RADAR_ASSOCIATION_MODES:
        raise ValueError(f"unknown radar association mode {association!r}")
    if association == "oracle-nearest-truth" and truth is None:
        raise ValueError("oracle-nearest-truth association requires normalized truth")
    if track_switch_nis_ratio <= 0.0:
        raise ValueError("track_switch_nis_ratio must be positive")

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


def _radar_row_to_measurement(row: pd.Series, covariance: np.ndarray) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=float(row["time_s"]),
        vector=np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])]),
        covariance=covariance,
        source="radar",
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
    for column in ("association_mode", "association_nis", "association_candidate_rows"):
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
