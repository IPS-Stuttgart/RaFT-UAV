"""IMM-native radar association for RaFT-UAV.

The legacy IMM command preselects radar rows, converts the selected rows into a
single measurement stream, and only then runs the IMM tracker.  That prevents
radar association from using the current IMM prediction and mode probabilities.

This module keeps association inside the IMM loop.  Radar candidates are scored
against the live IMM prediction; the selected row is then passed through the same
gating and robust-update machinery as ordinary RF/radar measurements.  The
implementation intentionally keeps the public surface small so it can be used as
one extra SOTA row without perturbing the existing IMM baseline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker
from raft_uav.baselines.kalman import TrackingMeasurement, measurement_matrix
from raft_uav.baselines.radar_association import (
    _catprob_candidate_pool,
    _empty_selected_radar,
    _events,
    _gate_threshold_for_measurement,
    _inflation_alpha_for_measurement,
    _initial_measurement_and_row,
    _max_residual_norm_for_measurement,
    _optional_float,
    _optional_track_id,
    _radar_row_to_measurement,
    _robust_update_for_measurement,
    _row_covariance,
    _selected_rows_frame,
)

IMM_RADAR_ASSOCIATION_MODES = (
    "imm-mixture-nis",
    "imm-rf-anchored-nis",
)


@dataclass(frozen=True)
class _CandidateScore:
    score: float
    nis: float
    log_likelihood: float
    best_mode: str | None


def run_async_imm_baseline_with_radar_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    association: str = "imm-mixture-nis",
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    candidate_catprob_threshold: float | None = 0.5,
    mode_switch_time_constant_s: float = 20.0,
    rf_anchor_weight: float = 0.35,
    rf_anchor_time_gate_s: float = 2.0,
    rf_anchor_nis_cap: float = 25.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run an IMM tracker while selecting at most one radar row per radar frame.

    ``imm-mixture-nis`` ranks candidates by their negative IMM mixture
    likelihood.  ``imm-rf-anchored-nis`` adds a bounded penalty from the latest
    causal RF measurement, which is useful when the IMM prediction is broad after
    a gap or maneuver.
    """

    if association not in IMM_RADAR_ASSOCIATION_MODES:
        raise ValueError(f"unknown IMM radar association mode {association!r}")
    if radar_xy_std_m <= 0.0 or radar_z_std_m <= 0.0:
        raise ValueError("radar_xy_std_m and radar_z_std_m must be positive")
    if rf_anchor_weight < 0.0:
        raise ValueError("rf_anchor_weight must be nonnegative")
    if rf_anchor_time_gate_s < 0.0:
        raise ValueError("rf_anchor_time_gate_s must be nonnegative")
    if rf_anchor_nis_cap <= 0.0:
        raise ValueError("rf_anchor_nis_cap must be positive")

    rf_measurement_list = list(rf_measurements)
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(rf_measurement_list, radar)
    if not events:
        return [], _empty_selected_radar(radar)

    initial_measurement: TrackingMeasurement | None = None
    initial_selected_row: pd.Series | None = None
    start_index = 0
    for index, event in enumerate(events):
        initial = _initial_measurement_and_row(
            event,
            association=association,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            truth=None,
            truth_gate_m=150.0,
            truth_time_gate_s=1.0,
        )
        if initial is None:
            continue
        initial_measurement, initial_selected_row = initial
        start_index = int(index)
        break

    if initial_measurement is None:
        return [], _empty_selected_radar(radar)

    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
        mode_switch_time_constant_s=mode_switch_time_constant_s,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []

    initial_diagnostics = tracker.update(
        initial_measurement,
        gate_threshold=_gate_threshold_for_measurement(
            initial_measurement,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
        ),
        safety_gate_threshold=_gate_threshold_for_measurement(
            initial_measurement,
            gate_probabilities_by_source=safety_gate_probabilities_by_source,
            gate_thresholds_by_source=safety_gate_thresholds_by_source,
        ),
        max_residual_norm=_max_residual_norm_for_measurement(
            initial_measurement,
            max_residual_norms_by_source=max_residual_norms_by_source,
        ),
        robust_update=_robust_update_for_measurement(
            initial_measurement,
            robust_update_by_source=robust_update_by_source,
        ),
        inflation_alpha=_inflation_alpha_for_measurement(
            initial_measurement,
            inflation_alpha_by_source=inflation_alpha_by_source,
        ),
    )
    if initial_selected_row is not None and initial_diagnostics.accepted:
        selected = initial_selected_row.copy()
        selected["association_mode"] = association
        selected["association_action"] = "imm_bootstrap"
        selected_rows.append(selected)
    records.append(
        _imm_record(
            initial_measurement, tracker, initial_diagnostics, initial_selected_row, association
        )
    )

    for event in events[start_index + 1 :]:
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
                safety_gate_threshold=_gate_threshold_for_measurement(
                    measurement,
                    gate_probabilities_by_source=safety_gate_probabilities_by_source,
                    gate_thresholds_by_source=safety_gate_thresholds_by_source,
                ),
                max_residual_norm=_max_residual_norm_for_measurement(
                    measurement,
                    max_residual_norms_by_source=max_residual_norms_by_source,
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
            records.append(_imm_record(measurement, tracker, diagnostics, None, association))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        time_s = float(event["time_s"])
        tracker.predict_to(time_s)
        selected = _select_imm_radar_candidate(
            candidates,
            tracker=tracker,
            base_covariance=covariance,
            association=association,
            candidate_catprob_threshold=candidate_catprob_threshold,
            rf_measurements=rf_measurement_list,
            rf_anchor_weight=rf_anchor_weight,
            rf_anchor_time_gate_s=rf_anchor_time_gate_s,
            rf_anchor_nis_cap=rf_anchor_nis_cap,
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
            safety_gate_threshold=_gate_threshold_for_measurement(
                measurement,
                gate_probabilities_by_source=safety_gate_probabilities_by_source,
                gate_thresholds_by_source=safety_gate_thresholds_by_source,
            ),
            max_residual_norm=_max_residual_norm_for_measurement(
                measurement,
                max_residual_norms_by_source=max_residual_norms_by_source,
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
            selected_rows.append(selected)
        records.append(_imm_record(measurement, tracker, diagnostics, selected, association))

    return records, _selected_rows_frame(radar, selected_rows)


def _select_imm_radar_candidate(
    candidates: pd.DataFrame,
    *,
    tracker: AsyncInteractingMultipleModelTracker,
    base_covariance: np.ndarray,
    association: str,
    candidate_catprob_threshold: float | None,
    rf_measurements: list[TrackingMeasurement],
    rf_anchor_weight: float,
    rf_anchor_time_gate_s: float,
    rf_anchor_nis_cap: float,
) -> pd.Series | None:
    pool = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    if pool.empty:
        return None
    scored = _imm_scored_candidates(pool, tracker=tracker, base_covariance=base_covariance)
    if association == "imm-rf-anchored-nis":
        scored = _add_rf_anchor_penalty(
            scored,
            rf_measurements=rf_measurements,
            anchor_weight=rf_anchor_weight,
            time_gate_s=rf_anchor_time_gate_s,
            nis_cap=rf_anchor_nis_cap,
        )
    best = scored.loc[scored["association_score"].idxmin()].copy()
    best["association_mode"] = association
    best["association_action"] = association.replace("-", "_")
    return best


def _imm_scored_candidates(
    candidates: pd.DataFrame,
    *,
    tracker: AsyncInteractingMultipleModelTracker,
    base_covariance: np.ndarray,
) -> pd.DataFrame:
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    covariances = _candidate_covariances(candidates, base_covariance)
    rows = candidates.copy()
    scores: list[_CandidateScore] = []
    for vector, covariance in zip(vectors, covariances, strict=True):
        scores.append(_candidate_score(vector, covariance, tracker=tracker))
    rows["association_nis"] = [score.nis for score in scores]
    rows["association_score"] = [score.score for score in scores]
    rows["association_imm_log_likelihood"] = [score.log_likelihood for score in scores]
    rows["association_imm_best_mode"] = [score.best_mode for score in scores]
    rows["association_candidate_rows"] = int(len(rows))
    return rows


def _candidate_score(
    vector: np.ndarray,
    measurement_covariance: np.ndarray,
    *,
    tracker: AsyncInteractingMultipleModelTracker,
) -> _CandidateScore:
    observation = measurement_matrix(3)
    residual = np.asarray(vector, dtype=float).reshape(3) - observation @ tracker.state
    combined_s = observation @ tracker.covariance_matrix @ observation.T + measurement_covariance
    combined_nis = _quadratic_form(residual, combined_s)

    mode_terms = _mode_log_likelihood_terms(vector, measurement_covariance, tracker=tracker)
    if not mode_terms:
        log_likelihood = _log_gaussian(residual, combined_s)
        return _CandidateScore(
            score=float(-2.0 * log_likelihood),
            nis=float(combined_nis),
            log_likelihood=float(log_likelihood),
            best_mode=None,
        )
    log_values = np.array([item[1] for item in mode_terms], dtype=float)
    log_likelihood = _logsumexp(log_values)
    best_mode = mode_terms[int(np.argmax(log_values))][0]
    return _CandidateScore(
        score=float(-2.0 * log_likelihood),
        nis=float(combined_nis),
        log_likelihood=float(log_likelihood),
        best_mode=best_mode,
    )


def _mode_log_likelihood_terms(
    vector: np.ndarray,
    measurement_covariance: np.ndarray,
    *,
    tracker: AsyncInteractingMultipleModelTracker,
) -> list[tuple[str, float]]:
    filters = _imm_subfilters(tracker)
    probabilities = np.asarray(tracker.mode_probabilities, dtype=float).reshape(-1)
    if not filters or len(filters) != len(probabilities):
        return []
    observation = measurement_matrix(3)
    terms: list[tuple[str, float]] = []
    for mode_name, probability, filter_obj in zip(
        tracker.mode_names,
        probabilities,
        filters,
        strict=False,
    ):
        if probability <= 0.0 or not np.isfinite(probability):
            continue
        try:
            state = np.asarray(filter_obj.get_point_estimate(), dtype=float).reshape(6)
            covariance = np.asarray(filter_obj.filter_state.C, dtype=float).reshape(6, 6)
        except Exception:
            continue
        residual = np.asarray(vector, dtype=float).reshape(3) - observation @ state
        innovation = observation @ covariance @ observation.T + measurement_covariance
        log_likelihood = float(
            np.log(probability) + _log_gaussian(residual, innovation)
        )
        terms.append((str(mode_name), log_likelihood))
    return terms


def _imm_subfilters(tracker: AsyncInteractingMultipleModelTracker) -> list[Any]:
    imm_filter = tracker.filter
    for attr in ("filters", "filter_bank", "filter_list", "_filters"):
        value = getattr(imm_filter, attr, None)
        if value is not None:
            return list(value)
    return []


def _candidate_covariances(
    candidates: pd.DataFrame,
    base_covariance: np.ndarray,
) -> list[np.ndarray]:
    base = np.asarray(base_covariance, dtype=float).reshape(3, 3)
    covariances: list[np.ndarray] = []
    for _, row in candidates.iterrows():
        row_covariance = _row_covariance(row)
        covariances.append(base if row_covariance is None else row_covariance)
    return covariances


def _add_rf_anchor_penalty(
    scored: pd.DataFrame,
    *,
    rf_measurements: list[TrackingMeasurement],
    anchor_weight: float,
    time_gate_s: float,
    nis_cap: float,
) -> pd.DataFrame:
    out = scored.copy()
    out["association_anchor_penalty"] = 0.0
    if anchor_weight <= 0.0 or not rf_measurements:
        return out
    time_s = float(out["time_s"].median())
    anchor = _latest_rf_anchor(rf_measurements, time_s=time_s, time_gate_s=time_gate_s)
    if anchor is None or anchor.vector.size < 2 or anchor.covariance.shape[0] < 2:
        return out
    covariance = np.asarray(anchor.covariance[:2, :2], dtype=float)
    try:
        precision = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(covariance)
    residuals = out[["east_m", "north_m"]].to_numpy(dtype=float)
    residuals = residuals - np.asarray(anchor.vector[:2], dtype=float)
    anchor_nis = np.einsum("ij,jk,ik->i", residuals, precision, residuals)
    anchor_nis = np.where(np.isfinite(anchor_nis), anchor_nis, np.inf)
    penalty = float(anchor_weight) * np.minimum(anchor_nis, float(nis_cap))
    out["association_anchor_nis"] = anchor_nis
    out["association_anchor_penalty"] = penalty
    out["association_anchor_time_delta_s"] = float(time_s - anchor.time_s)
    out["association_score"] = out["association_score"].to_numpy(dtype=float) + penalty
    return out


def _latest_rf_anchor(
    rf_measurements: list[TrackingMeasurement],
    *,
    time_s: float,
    time_gate_s: float,
) -> TrackingMeasurement | None:
    best: TrackingMeasurement | None = None
    best_dt_s = float("inf")
    for measurement in rf_measurements:
        if measurement.source != "rf":
            continue
        dt_s = float(time_s) - float(measurement.time_s)
        if dt_s < -1.0e-9 or dt_s > float(time_gate_s):
            continue
        if dt_s < best_dt_s:
            best = measurement
            best_dt_s = dt_s
    return best


def _imm_record(
    measurement: TrackingMeasurement,
    tracker: AsyncInteractingMultipleModelTracker,
    diagnostics: Any,
    selected: pd.Series | None,
    association: str,
) -> dict[str, object]:
    record: dict[str, object] = {
        "time_s": measurement.time_s,
        "source": measurement.source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        "mode_names": tracker.mode_names,
        "mode_probabilities": tracker.mode_probabilities.copy(),
        "mode_probability_map": tracker.mode_probability_map,
        "most_likely_mode": tracker.most_likely_mode_name,
        **diagnostics.to_record(),
    }
    if selected is not None:
        record["track_id"] = _optional_track_id(selected)
        record["association_mode"] = str(selected.get("association_mode", association))
        record["association_nis"] = _optional_float(selected.get("association_nis"))
        record["association_score"] = _optional_float(selected.get("association_score"))
        best_mode = selected.get("association_imm_best_mode")
        if best_mode is not None and not pd.isna(best_mode):
            record["association_imm_best_mode"] = str(best_mode)
    return record


def _quadratic_form(residual: np.ndarray, covariance: np.ndarray) -> float:
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    value = float(np.asarray(residual, dtype=float).T @ solved)
    return value if np.isfinite(value) else float("inf")


def _log_gaussian(residual: np.ndarray, covariance: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    jitter = 1.0e-9 * max(float(np.trace(covariance)), 1.0)
    covariance = covariance + np.eye(covariance.shape[0]) * jitter
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        covariance = covariance + np.eye(covariance.shape[0]) * max(jitter, 1.0e-6)
        sign, logdet = np.linalg.slogdet(covariance)
    nis = _quadratic_form(residual, covariance)
    dim = residual.size
    return float(-0.5 * (nis + logdet + dim * np.log(2.0 * np.pi)))


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("-inf")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))
