"""Radar row association for the asynchronous CV baseline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd
from pyrecest.filters import KalmanFilter
from pyrecest.filters.multi_hypothesis_tracker import MultiHypothesisTracker

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
    TrackingUpdateDiagnostics,
    constant_velocity_matrix,
    gate_threshold_from_probability,
    measurement_matrix,
    white_acceleration_process_noise,
)
from raft_uav.baselines.radar_update_policy import (
    apply_radar_update_policy,
    policy_record_fields,
)
from raft_uav.baselines.update_logic import (
    max_residual_norm_for_measurement,
    plan_linear_measurement_update,
)
from raft_uav.numeric import optional_float, optional_int

RadarCovarianceFn = Callable[[pd.Series, np.ndarray], np.ndarray]

RADAR_ASSOCIATION_MODES = (
    "oracle-nearest-truth",
    "prediction-nis",
    "rf-anchored-nis",
    "rf-gated-nis",
    "track-continuity",
    "geometry-score",
    "pda-mixture",
    "track-bank",
    "paper-compatible",
    "paper-largest-continuous-track",
    "stable-segments",
    "stable-segments-hybrid",
    "stable-segments-interpolated",
)
_STABLE_SEGMENT_ASSOCIATION_MODES = {
    "paper-largest-continuous-track",
    "stable-segments",
    "stable-segments-interpolated",
}
_STABLE_SEGMENT_PRECOMPUTE_MODES = {
    *_STABLE_SEGMENT_ASSOCIATION_MODES,
    "stable-segments-hybrid",
}
_SOFT_CATPROB_RETENTION_ENV = "RAFT_UAV_SOFT_CATPROB_RETENTION"
_SOFT_CATPROB_BELOW_THRESHOLD_PENALTY_ENV = "RAFT_UAV_SOFT_CATPROB_BELOW_THRESHOLD_PENALTY"
_RADAR_VELOCITY_UPDATE_ENV = "RAFT_UAV_RADAR_UPDATE_USES_VELOCITY"
_RADAR_VELOCITY_STD_MPS_ENV = "RAFT_UAV_RADAR_VELOCITY_STD_MPS"
RADAR_COVARIANCE_MODELS = ("cartesian", "geometry")
PAPER_COMPATIBLE_NIS_GATE_PROBABILITY = 0.95
_ASSOCIATION_COVARIANCE_COLUMNS = (
    "association_cov_ee",
    "association_cov_nn",
    "association_cov_uu",
    "association_cov_en",
    "association_cov_eu",
    "association_cov_nu",
)


@dataclass(frozen=True)
class _RadarGeometryCovarianceConfig:
    model: str = "cartesian"
    radial_std_m: float = 12.0
    radial_range_fraction: float = 0.005
    crossrange_angle_std_deg: float = 1.5
    crossrange_min_std_m: float = 5.0
    crossrange_max_std_m: float = 80.0
    min_std_m: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.model == "geometry"

    @property
    def crossrange_angle_std_rad(self) -> float:
        return float(np.deg2rad(self.crossrange_angle_std_deg))


@dataclass(frozen=True)
class _TrackSegment:
    frame: pd.DataFrame
    track_id: int
    start_time_s: float
    end_time_s: float
    start_position_m: np.ndarray
    end_position_m: np.ndarray
    frames: int
    mean_catprob: float
    rf_support_count: int = 0
    rf_mean_nis: float | None = None
    rf_score_adjustment: float = 0.0

    @property
    def score(self) -> float:
        return float(self.frames) * max(self.mean_catprob, 0.0) + float(
            self.rf_score_adjustment
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
    radar_covariance_model: str = "cartesian",
    radar_range_std_m: float = 12.0,
    radar_range_std_fraction: float = 0.005,
    radar_crossrange_angle_std_deg: float = 1.5,
    radar_crossrange_min_std_m: float = 5.0,
    radar_crossrange_max_std_m: float = 80.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
    track_switch_nis_ratio: float = 0.5,
    candidate_catprob_threshold: float | None = 0.5,
    geometry_velocity_std_mps: float = 12.0,
    geometry_velocity_weight: float = 0.25,
    geometry_switch_penalty: float = 4.0,
    geometry_catprob_weight: float = 2.0,
    rf_anchor_weight: float = 0.35,
    rf_anchor_time_gate_s: float = 2.0,
    rf_anchor_nis_cap: float = 25.0,
    rf_anchor_gate_nis: float = 25.0,
    pda_nis_temperature: float = 1.0,
    pda_catprob_exponent: float = 1.0,
    track_bank_max_hypotheses: int = 16,
    track_bank_max_assignments: int = 16,
    track_bank_max_candidates: int = 16,
    track_bank_gate_probability: float = 0.9999999,
    track_bank_detection_probability: float = 0.999,
    track_bank_clutter_intensity: float = 1.0e-12,
    track_bank_prune_log_weight_delta: float = 80.0,
    stable_segment_min_frames: int = 100,
    stable_segment_max_transition_speed_mps: float = 65.0,
    stable_segment_range_gate_m: float | None = 800.0,
    stable_segment_interpolation_max_gap_s: float | None = 5.0,
    stable_segment_interpolation_max_speed_mps: float | None = 65.0,
    stable_segment_interpolation_std_scale: float = 2.0,
    stable_segment_interpolation_gap_std_mps: float = 12.0,
    stable_segment_rf_score_weight: float = 1.0,
    stable_segment_rf_time_gate_s: float = 2.0,
    stable_segment_rf_nis_cap: float = 25.0,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion while selecting at most one radar row per radar frame.

    ``oracle-nearest-truth`` uses ground truth and is only a diagnostic upper
    bound. ``prediction-nis`` picks the radar candidate with the lowest
    normalized innovation squared against the current predicted state.
    ``rf-anchored-nis`` adds a capped RF-position penalty from the most recent
    nearby RF update, which helps recover when the prediction has drifted.
    ``rf-gated-nis`` uses the same RF anchor but drops radar frames whose
    candidates are all inconsistent with the recent RF position.
    ``track-continuity`` prefers the current Fortem track ID and switches only
    when another candidate has a substantially lower NIS. ``geometry-score``
    is an online score that augments NIS with radar velocity consistency,
    track switching, and UAV class-probability penalties. ``pda-mixture``
    keeps a single Kalman update but forms it from a probability-weighted
    candidate mixture and adds candidate spread to the radar covariance.
    ``track-bank`` uses PyRecEst's track-oriented MHT to keep multiple
    single-target association hypotheses alive across radar frames.
    ``paper-compatible`` applies the reference-style hard preselector directly
    in the baseline runner: 800 m-style range gating via
    ``stable_segment_range_gate_m``, largest continuous Fortem track retention,
    hard UAV class-probability thresholding, NIS validation, and explicit
    missed-detection coasting when no radar candidate passes.
    ``paper-largest-continuous-track`` applies the reference-paper style
    800 m range-gated longest continuous Fortem track preselector and skips
    all other radar frames.
    ``stable-segments`` preselects stitched high-confidence Fortem track
    segments and skips all other radar frames. ``stable-segments-hybrid``
    uses those anchors when available and falls back to prediction-NIS
    otherwise. ``stable-segments-interpolated`` uses the same anchors, then
    fills only radar frame times bracketed by plausible stable anchors.
    """

    if association not in RADAR_ASSOCIATION_MODES:
        raise ValueError(f"unknown radar association mode {association!r}")
    if radar_covariance_model not in RADAR_COVARIANCE_MODELS:
        raise ValueError(f"unknown radar covariance model {radar_covariance_model!r}")
    if radar_xy_std_m <= 0.0 or radar_z_std_m <= 0.0:
        raise ValueError("radar_xy_std_m and radar_z_std_m must be positive")
    if radar_range_std_m <= 0.0:
        raise ValueError("radar_range_std_m must be positive")
    if radar_range_std_fraction < 0.0:
        raise ValueError("radar_range_std_fraction must be nonnegative")
    if radar_crossrange_angle_std_deg <= 0.0:
        raise ValueError("radar_crossrange_angle_std_deg must be positive")
    if radar_crossrange_min_std_m <= 0.0:
        raise ValueError("radar_crossrange_min_std_m must be positive")
    if radar_crossrange_max_std_m < radar_crossrange_min_std_m:
        raise ValueError("radar_crossrange_max_std_m must be >= radar_crossrange_min_std_m")
    if association == "oracle-nearest-truth" and truth is None:
        raise ValueError("oracle-nearest-truth association requires normalized truth")
    if track_switch_nis_ratio <= 0.0:
        raise ValueError("track_switch_nis_ratio must be positive")
    if geometry_velocity_std_mps <= 0.0:
        raise ValueError("geometry_velocity_std_mps must be positive")
    if rf_anchor_weight < 0.0:
        raise ValueError("rf_anchor_weight must be nonnegative")
    if rf_anchor_time_gate_s < 0.0:
        raise ValueError("rf_anchor_time_gate_s must be nonnegative")
    if rf_anchor_nis_cap <= 0.0:
        raise ValueError("rf_anchor_nis_cap must be positive")
    if rf_anchor_gate_nis <= 0.0:
        raise ValueError("rf_anchor_gate_nis must be positive")
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
    if track_bank_max_hypotheses < 1:
        raise ValueError("track_bank_max_hypotheses must be positive")
    if track_bank_max_assignments < 1:
        raise ValueError("track_bank_max_assignments must be positive")
    if track_bank_max_candidates < 1:
        raise ValueError("track_bank_max_candidates must be positive")
    if not 0.0 < track_bank_gate_probability < 1.0:
        raise ValueError("track_bank_gate_probability must be in (0, 1)")
    if not 0.0 < track_bank_detection_probability < 1.0:
        raise ValueError("track_bank_detection_probability must be in (0, 1)")
    if track_bank_clutter_intensity <= 0.0:
        raise ValueError("track_bank_clutter_intensity must be positive")
    if track_bank_prune_log_weight_delta <= 0.0:
        raise ValueError("track_bank_prune_log_weight_delta must be positive")
    if stable_segment_min_frames < 1:
        raise ValueError("stable_segment_min_frames must be positive")
    if stable_segment_max_transition_speed_mps <= 0.0:
        raise ValueError("stable_segment_max_transition_speed_mps must be positive")
    if stable_segment_range_gate_m is not None and stable_segment_range_gate_m <= 0.0:
        raise ValueError("stable_segment_range_gate_m must be positive or None")
    if (
        stable_segment_interpolation_max_gap_s is not None
        and stable_segment_interpolation_max_gap_s <= 0.0
    ):
        raise ValueError("stable_segment_interpolation_max_gap_s must be positive or None")
    if (
        stable_segment_interpolation_max_speed_mps is not None
        and stable_segment_interpolation_max_speed_mps <= 0.0
    ):
        raise ValueError("stable_segment_interpolation_max_speed_mps must be positive or None")
    if stable_segment_interpolation_std_scale <= 0.0:
        raise ValueError("stable_segment_interpolation_std_scale must be positive")
    if stable_segment_interpolation_gap_std_mps < 0.0:
        raise ValueError("stable_segment_interpolation_gap_std_mps must be nonnegative")
    if stable_segment_rf_score_weight < 0.0:
        raise ValueError("stable_segment_rf_score_weight must be nonnegative")
    if stable_segment_rf_time_gate_s < 0.0:
        raise ValueError("stable_segment_rf_time_gate_s must be nonnegative")
    if stable_segment_rf_nis_cap <= 0.0:
        raise ValueError("stable_segment_rf_nis_cap must be positive")

    rf_measurement_list = list(rf_measurements)
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    _validate_normalized_radar_for_association(radar)
    covariance_config = _RadarGeometryCovarianceConfig(
        model=radar_covariance_model,
        radial_std_m=float(radar_range_std_m),
        radial_range_fraction=float(radar_range_std_fraction),
        crossrange_angle_std_deg=float(radar_crossrange_angle_std_deg),
        crossrange_min_std_m=float(radar_crossrange_min_std_m),
        crossrange_max_std_m=float(radar_crossrange_max_std_m),
    )
    if association == "paper-compatible":
        return _run_paper_compatible_association(
            rf_measurements=rf_measurement_list,
            radar=radar,
            covariance=covariance,
            covariance_config=covariance_config,
            acceleration_std_mps2=acceleration_std_mps2,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            candidate_catprob_threshold=candidate_catprob_threshold,
            range_gate_m=stable_segment_range_gate_m,
            track_switch_penalty=geometry_switch_penalty,
            catprob_weight=geometry_catprob_weight,
        )
    if association == "track-bank":
        _validate_normalized_radar_for_association(radar)
        return _run_mht_track_bank(
            rf_measurements=rf_measurement_list,
            radar=radar,
            covariance=covariance,
            covariance_config=covariance_config,
            acceleration_std_mps2=acceleration_std_mps2,
            gate_probabilities_by_source=gate_probabilities_by_source,
            gate_thresholds_by_source=gate_thresholds_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
            candidate_catprob_threshold=candidate_catprob_threshold,
            max_global_hypotheses=track_bank_max_hypotheses,
            max_assignments_per_hypothesis=track_bank_max_assignments,
            max_candidates_per_track=track_bank_max_candidates,
            gate_probability=track_bank_gate_probability,
            detection_probability=track_bank_detection_probability,
            clutter_intensity=track_bank_clutter_intensity,
            prune_log_weight_delta=track_bank_prune_log_weight_delta,
        )

    events = _events(rf_measurement_list, radar)
    if not events:
        return [], _empty_selected_radar(radar)

    stable_anchor_by_key: dict[object, pd.Series] | None = None
    if association in _STABLE_SEGMENT_PRECOMPUTE_MODES:
        if association == "paper-largest-continuous-track":
            stable_anchors = _select_largest_continuous_radar_track(
                radar,
                range_gate_m=stable_segment_range_gate_m,
            )
        else:
            stable_anchors = _select_stable_radar_segments(
                radar,
                range_gate_m=stable_segment_range_gate_m,
                catprob_threshold=candidate_catprob_threshold,
                min_segment_frames=stable_segment_min_frames,
                max_transition_speed_mps=stable_segment_max_transition_speed_mps,
                rf_measurements=rf_measurement_list,
                rf_score_weight=stable_segment_rf_score_weight,
                rf_time_gate_s=stable_segment_rf_time_gate_s,
                rf_nis_cap=stable_segment_rf_nis_cap,
            )
        if association == "stable-segments-interpolated":
            stable_anchors = _interpolate_stable_radar_segments_to_frame_times(
                radar,
                stable_anchors,
                association_mode=association,
                base_covariance=covariance,
                max_gap_s=stable_segment_interpolation_max_gap_s,
                max_speed_mps=stable_segment_interpolation_max_speed_mps,
                interpolated_std_scale=stable_segment_interpolation_std_scale,
                gap_std_mps=stable_segment_interpolation_gap_std_mps,
            )
        stable_anchor_by_key = {
            _radar_row_key(row): row for _, row in stable_anchors.iterrows()
        }

    start_index = 0
    initial_measurement = None
    initial_selected_row: pd.Series | None = None
    initial_events = (
        enumerate(events) if association in _STABLE_SEGMENT_ASSOCIATION_MODES else [(0, events[0])]
    )
    for index, event in initial_events:
        initial = _initial_measurement_and_row(
            event,
            association=association,
            covariance=covariance,
            covariance_config=covariance_config,
            stable_anchor_by_key=stable_anchor_by_key,
            candidate_catprob_threshold=candidate_catprob_threshold,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        if initial is not None:
            initial_measurement, initial_selected_row = initial
            start_index = int(index)
            break
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
        current_track_id = _optional_track_id(initial_selected_row)
        selected_rows.append(initial_selected_row)
    records.append(
        _record(
            initial_measurement,
            tracker,
            initial_diagnostics,
            **_selected_row_record_kwargs(initial_selected_row, association),
        )
    )

    for event in events[start_index + 1:]:
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
            covariance_config=covariance_config,
            truth=truth,
            current_track_id=current_track_id,
            track_switch_nis_ratio=track_switch_nis_ratio,
            candidate_catprob_threshold=candidate_catprob_threshold,
            geometry_velocity_std_mps=geometry_velocity_std_mps,
            geometry_velocity_weight=geometry_velocity_weight,
            geometry_switch_penalty=geometry_switch_penalty,
            geometry_catprob_weight=geometry_catprob_weight,
            rf_measurements=rf_measurement_list,
            rf_anchor_weight=rf_anchor_weight,
            rf_anchor_time_gate_s=rf_anchor_time_gate_s,
            rf_anchor_nis_cap=rf_anchor_nis_cap,
            rf_anchor_gate_nis=rf_anchor_gate_nis,
            pda_nis_temperature=pda_nis_temperature,
            pda_catprob_exponent=pda_catprob_exponent,
            stable_anchor_by_key=stable_anchor_by_key,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        if selected is None:
            continue

        measurement = _radar_row_to_measurement(selected, covariance)
        selected, measurement, policy_diagnostics = apply_radar_update_policy(selected, measurement)
        diagnostics = policy_diagnostics or tracker.update(
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
            current_track_id = _optional_track_id(selected)
            selected_rows.append(selected)
        record = _record(
            measurement,
            tracker,
            diagnostics,
            track_id=_optional_track_id(selected),
            association_nis=_optional_float(selected.get("association_nis")),
            association_score=_optional_float(selected.get("association_score")),
            association_mode=association,
        )
        record.update(policy_record_fields(selected))
        records.append(record)

    return records, _selected_rows_frame(radar, selected_rows)


def _validate_normalized_radar_for_association(radar: pd.DataFrame) -> None:
    """Fail early when association receives raw radar rows instead of ENU rows."""

    if radar.empty:
        return
    required = {"time_s", "east_m", "north_m", "up_m"}
    missing = sorted(required.difference(radar.columns))
    if missing:
        raise ValueError(
            "radar association requires normalized radar rows with columns "
            f"{sorted(required)}; missing {missing}"
        )


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


def _run_paper_compatible_association(
    *,
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None,
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    candidate_catprob_threshold: float | None,
    range_gate_m: float | None,
    track_switch_penalty: float,
    catprob_weight: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run the paper-compatible hard-gated radar association inside the baseline.

    This mode intentionally differs from the exploratory association modes: it
    has no class-probability fallback, uses the range-gated largest continuous
    Fortem track as the radar preselector, and records a posterior at every
    radar frame.  Frames that fail range/class/NIS validation become explicit
    ``missed_detection`` records so the output timeline and metrics are
    comparable to the diagnostic paper-table implementation.
    """

    events = _events(rf_measurements, radar)
    if not events:
        return [], _empty_selected_radar(radar)

    paper_track = _select_paper_compatible_radar_track(
        radar,
        range_gate_m=range_gate_m,
        catprob_threshold=candidate_catprob_threshold,
    )
    paper_track_by_key = {_radar_row_key(row): row for _, row in paper_track.iterrows()}
    initial = _initial_paper_compatible_measurement_and_row(
        events,
        covariance=covariance,
        covariance_config=covariance_config,
        preselected_radar_by_key=paper_track_by_key,
        range_gate_m=range_gate_m,
        candidate_catprob_threshold=candidate_catprob_threshold,
    )
    if initial is None:
        return [], _empty_selected_radar(radar)
    start_index, initial_measurement, initial_selected_row = initial

    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    current_track_id: int | None = None

    initial_diagnostics = tracker.update(
        initial_measurement,
        gate_threshold=_paper_gate_threshold_for_measurement(
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
        current_track_id = _optional_track_id(initial_selected_row)
        selected_rows.append(initial_selected_row)
    records.append(
        _record(
            initial_measurement,
            tracker,
            initial_diagnostics,
            **_selected_row_record_kwargs(initial_selected_row, "paper-compatible"),
        )
    )

    radar_nis_threshold = _paper_source_gate_threshold(
        source="radar",
        measurement_dim=3,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
    )

    for event in events[start_index + 1:]:
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = tracker.update(
                measurement,
                gate_threshold=_paper_gate_threshold_for_measurement(
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
            records.append(_record(measurement, tracker, diagnostics))
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        tracker.predict_to(time_s)
        preselected_row = paper_track_by_key.get(_radar_event_key(event))
        selected = _select_paper_compatible_candidate(
            candidates,
            tracker=tracker,
            covariance=covariance,
            covariance_config=covariance_config,
            preselected_row=preselected_row,
            current_track_id=current_track_id,
            range_gate_m=range_gate_m,
            candidate_catprob_threshold=candidate_catprob_threshold,
            nis_gate_threshold=radar_nis_threshold,
            track_switch_penalty=track_switch_penalty,
            catprob_weight=catprob_weight,
        )
        if selected is None:
            records.append(
                _missed_detection_record(
                    time_s=time_s,
                    tracker=tracker,
                    source="radar",
                    association_mode="paper-compatible",
                    gate_threshold=radar_nis_threshold,
                )
            )
            continue

        measurement = _radar_row_to_measurement(selected, covariance)
        selected, measurement, policy_diagnostics = apply_radar_update_policy(selected, measurement)
        diagnostics = policy_diagnostics or tracker.update(
            measurement,
            gate_threshold=_paper_gate_threshold_for_measurement(
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
            current_track_id = _optional_track_id(selected)
            selected_rows.append(selected)
        record = _record(
            measurement,
            tracker,
            diagnostics,
            track_id=_optional_track_id(selected),
            association_nis=_optional_float(selected.get("association_nis")),
            association_score=_optional_float(selected.get("association_score")),
            association_mode="paper-compatible",
        )
        record.update(policy_record_fields(selected))
        records.append(record)

    return records, _selected_rows_frame(radar, selected_rows)


def _initial_paper_compatible_measurement_and_row(
    events: list[dict[str, object]],
    *,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None,
    preselected_radar_by_key: dict[object, pd.Series],
    range_gate_m: float | None,
    candidate_catprob_threshold: float | None,
) -> tuple[int, TrackingMeasurement, pd.Series | None] | None:
    """Return the first valid bootstrap event for paper-compatible fusion."""

    for index, event in enumerate(events):
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            return int(index), measurement, None
        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        selected_row = preselected_radar_by_key.get(_radar_event_key(event))
        if selected_row is None:
            continue
        selected = _select_paper_compatible_bootstrap_candidate(
            covariance=covariance,
            covariance_config=covariance_config,
            preselected_row=selected_row,
            range_gate_m=range_gate_m,
            candidate_catprob_threshold=candidate_catprob_threshold,
        )
        if selected is not None:
            return int(index), _radar_row_to_measurement(selected, covariance), selected
    return None


def _select_paper_compatible_bootstrap_candidate(
    *,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None,
    preselected_row: pd.Series,
    range_gate_m: float | None,
    candidate_catprob_threshold: float | None,
) -> pd.Series | None:
    pool = _paper_compatible_preselector_pool(
        pd.DataFrame([preselected_row.copy()]),
        range_gate_m=range_gate_m,
        candidate_catprob_threshold=candidate_catprob_threshold,
    )
    selected = _highest_catprob(pool)
    if selected is None:
        return None
    selected = selected.copy()
    selected["association_mode"] = "paper-compatible"
    selected["association_action"] = "paper_compatible_bootstrap"
    selected["association_effective_candidates"] = int(len(pool))
    return _annotate_radar_geometry_covariance(selected, covariance, covariance_config)


def _select_paper_compatible_candidate(
    candidates: pd.DataFrame,
    *,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None,
    preselected_row: pd.Series | None,
    current_track_id: int | None,
    range_gate_m: float | None,
    candidate_catprob_threshold: float | None,
    nis_gate_threshold: float | None,
    track_switch_penalty: float,
    catprob_weight: float,
) -> pd.Series | None:
    if preselected_row is None:
        return None
    pool = _paper_compatible_preselector_pool(
        pd.DataFrame([preselected_row.copy()]),
        range_gate_m=range_gate_m,
        candidate_catprob_threshold=candidate_catprob_threshold,
    )
    if pool.empty:
        return None

    scored = _nis_scored_candidates(
        pool,
        tracker,
        covariance,
        covariance_config=covariance_config,
    )
    if scored.empty:
        return None
    nis_values = pd.to_numeric(scored["association_nis"], errors="coerce")
    if nis_gate_threshold is None:
        kept = scored.copy()
        rejected_count = 0
    else:
        keep = nis_values <= float(nis_gate_threshold)
        rejected_count = int((~keep).sum())
        kept = scored.loc[keep].copy()
    if kept.empty:
        return None

    kept["association_track_switch_penalty"] = _track_switch_penalty(
        kept,
        current_track_id=current_track_id,
        switch_penalty=float(track_switch_penalty),
    )
    kept["association_catprob_penalty"] = _catprob_penalty(kept, catprob_weight)
    kept["association_score"] = (
        kept["association_nis"].to_numpy(dtype=float)
        + kept["association_track_switch_penalty"].to_numpy(dtype=float)
        + kept["association_catprob_penalty"].to_numpy(dtype=float)
    )
    selected = kept.loc[kept["association_score"].idxmin()].copy()
    selected["association_mode"] = "paper-compatible"
    selected["association_action"] = "hard_gated_update"
    selected["association_effective_candidates"] = int(len(kept))
    selected["association_nis_gate_threshold"] = (
        np.nan if nis_gate_threshold is None else float(nis_gate_threshold)
    )
    selected["association_nis_gate_rejected_count"] = int(rejected_count)
    return _annotate_radar_geometry_covariance(selected, covariance, covariance_config)


def _paper_compatible_preselector_pool(
    candidates: pd.DataFrame,
    *,
    range_gate_m: float | None,
    candidate_catprob_threshold: float | None,
) -> pd.DataFrame:
    """Apply hard range and catProb preselection."""

    raw_count = int(len(candidates))
    pool = _range_candidate_pool(
        candidates,
        range_gate_m,
        require_fortem_range=range_gate_m is not None,
        context="paper-compatible",
    )
    range_count = int(len(pool))
    pool = _catprob_candidate_pool(
        pool,
        candidate_catprob_threshold,
        fallback_to_unfiltered=False,
    )
    if not pool.empty:
        pool["association_preselector_raw_rows"] = _preselector_count(
            candidates,
            "association_preselector_raw_rows",
            raw_count,
        )
        pool["association_preselector_range_gated_rows"] = _preselector_count(
            candidates,
            "association_preselector_range_gated_rows",
            range_count,
        )
        pool["association_preselector_segment_rows"] = _preselector_count(
            candidates,
            "association_preselector_segment_rows",
            int(len(pool)),
        )
        pool["association_preselector_catprob_rows"] = _preselector_count(
            candidates,
            "association_preselector_catprob_rows",
            int(len(pool)),
        )
    return pool


def _preselector_count(candidates: pd.DataFrame, column: str, fallback: int) -> int:
    if column not in candidates.columns or candidates.empty:
        return int(fallback)
    values = pd.to_numeric(candidates[column], errors="coerce").dropna()
    if values.empty:
        return int(fallback)
    return int(values.iloc[0])


def _paper_compatible_track_id(radar: pd.DataFrame, *, range_gate_m: float | None) -> int | None:
    """Return the track ID of the largest continuous range-gated radar segment."""

    pool = _range_candidate_pool(
        radar,
        range_gate_m,
        require_fortem_range=range_gate_m is not None,
        context="paper-compatible",
    )
    if pool.empty or "track_id" not in pool.columns:
        return None
    segments = _stable_track_segments(
        pool,
        min_segment_frames=1,
        rf_measurements=[],
        rf_score_weight=0.0,
        rf_time_gate_s=0.0,
        rf_nis_cap=1.0,
    )
    if not segments:
        return None
    selected_segment = max(
        segments,
        key=lambda item: (
            item.frames,
            item.end_time_s - item.start_time_s,
            item.mean_catprob,
            -item.start_time_s,
            -item.track_id,
        ),
    )
    return int(selected_segment.track_id)


def _paper_gate_threshold_for_measurement(
    measurement: TrackingMeasurement,
    *,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
) -> float | None:
    return _paper_source_gate_threshold(
        source=measurement.source,
        measurement_dim=measurement.vector.size,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
    )


def _paper_source_gate_threshold(
    *,
    source: str,
    measurement_dim: int,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
) -> float | None:
    """Return explicit gates or the paper-compatible default 0.95 NIS gate."""

    if gate_thresholds_by_source and source in gate_thresholds_by_source:
        threshold = gate_thresholds_by_source[source]
        return None if threshold is None else float(threshold)
    if gate_probabilities_by_source and source in gate_probabilities_by_source:
        return gate_threshold_from_probability(
            gate_probabilities_by_source[source],
            measurement_dim,
        )
    if source in {"rf", "radar"}:
        return gate_threshold_from_probability(
            PAPER_COMPATIBLE_NIS_GATE_PROBABILITY,
            measurement_dim,
        )
    return None


def _missed_detection_record(
    *,
    time_s: float,
    tracker: AsyncConstantVelocityKalmanTracker,
    source: str,
    association_mode: str,
    gate_threshold: float | None,
) -> dict[str, object]:
    diagnostics = TrackingUpdateDiagnostics(
        time_s=float(time_s),
        source=source,
        measurement_dim=3,
        accepted=False,
        update_action="missed_detection",
        nis=float("nan"),
        gate_threshold=gate_threshold,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )
    return {
        "time_s": float(time_s),
        "source": source,
        "state": tracker.state.copy(),
        "covariance": tracker.covariance_matrix.copy(),
        **diagnostics.to_record(),
        "association_mode": association_mode,
    }


def _first_rf_bootstrap_index(events: list[dict[str, object]]) -> int | None:
    """Return the preferred causal bootstrap point for association modes with RF support.

    If RF measurements are present, skip pre-RF radar frames rather than seeding
    a single-target track bank from an arbitrary class-probability radar row.
    Radar-only inputs retain the historical first-event bootstrap.
    """

    for index, event in enumerate(events):
        if event.get("kind") == "rf":
            return int(index)
    return 0 if events else None


def _run_mht_track_bank(
    *,
    rf_measurements: list[TrackingMeasurement],
    radar: pd.DataFrame,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None = None,
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    candidate_catprob_threshold: float | None,
    max_global_hypotheses: int,
    max_assignments_per_hypothesis: int,
    max_candidates_per_track: int,
    gate_probability: float,
    detection_probability: float,
    clutter_intensity: float,
    prune_log_weight_delta: float,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    events = _events(rf_measurements, radar)
    if not events:
        return [], _empty_selected_radar(radar)
    initial_event_index = _first_rf_bootstrap_index(events)
    if initial_event_index is None:
        return [], _empty_selected_radar(radar)
    events = events[initial_event_index:]

    initial_event = events[0]
    initial = _initial_measurement_and_row(
        initial_event,
        association="track-bank",
        covariance=covariance,
        covariance_config=covariance_config,
        stable_anchor_by_key=None,
        candidate_catprob_threshold=candidate_catprob_threshold,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial is None:
        return [], _empty_selected_radar(radar)
    initial_measurement, initial_selected_row = initial

    tracker = _initial_mht_tracker(
        initial_measurement,
        max_global_hypotheses=max_global_hypotheses,
        max_assignments_per_hypothesis=max_assignments_per_hypothesis,
        max_candidates_per_track=max_candidates_per_track,
        gate_probability=gate_probability,
        detection_probability=detection_probability,
        clutter_intensity=clutter_intensity,
        prune_log_weight_delta=prune_log_weight_delta,
    )
    current_time_s = float(initial_measurement.time_s)
    records: list[dict[str, object]] = []
    selected_rows: list[pd.Series] = []
    initial_diagnostics = _mht_bootstrap_diagnostics(
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
    )
    if initial_selected_row is not None:
        initial_selected_row = initial_selected_row.copy()
        initial_selected_row["association_mode"] = "track-bank"
        selected_rows.append(initial_selected_row)
    if initial_event["kind"] != "rf":
        initial_record_kwargs = _selected_row_record_kwargs(initial_selected_row, "track-bank")
        initial_record_kwargs.setdefault("association_mode", "track-bank")
        records.append(
            _mht_record(
                measurement=initial_measurement,
                tracker=tracker,
                diagnostics=initial_diagnostics,
                **initial_record_kwargs,
            )
        )

    # The first event was already assimilated into the MHT prior above.
    # Replaying it here would double-count the bootstrap measurement.
    for event in events[1:]:
        time_s = float(event["time_s"])
        _predict_mht_to(
            tracker,
            current_time_s=current_time_s,
            target_time_s=time_s,
            acceleration_std_mps2=acceleration_std_mps2,
        )
        current_time_s = time_s

        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            diagnostics = _deterministic_update_mht_hypotheses(
                tracker,
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
            records.append(
                _mht_record(
                    measurement=measurement,
                    tracker=tracker,
                    diagnostics=diagnostics,
                    association_mode="track-bank",
                )
            )
            continue

        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        candidates = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
        if candidates.empty:
            continue

        pre_update_nis = _nis_scored_candidates(
            candidates,
            _tracker_from_best_mht_hypothesis(tracker, current_time_s, acceleration_std_mps2),
            covariance,
            covariance_config=covariance_config,
        )
        measurements = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float).T
        covariances = _candidate_covariance_tensor(pre_update_nis, covariance)
        tracker.update_linear(measurements, measurement_matrix(3), covariances)

        selected = _selected_row_from_best_mht_assignment(
            candidates,
            pre_update_nis,
            tracker,
        )
        if selected is not None:
            selected_rows.append(selected)

        measurement = _mht_radar_measurement(time_s=time_s, selected=selected, tracker=tracker)
        records.append(
            _mht_record(
                measurement=measurement,
                tracker=tracker,
                diagnostics=_mht_radar_diagnostics(time_s, selected),
                track_id=None if selected is None else _optional_track_id(selected),
                association_nis=None if selected is None else _optional_float(selected.get("association_nis")),
                association_score=None
                if selected is None
                else _optional_float(selected.get("association_score")),
                association_mode="track-bank",
            )
        )

    return records, _selected_rows_frame(radar, selected_rows)


class _MHTStateView:
    def __init__(self, filter_obj: KalmanFilter) -> None:
        self._filter = filter_obj

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self._filter.get_point_estimate(), dtype=float).reshape(6)

    @property
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(self._filter.filter_state.C, dtype=float).reshape(6, 6)


def _initial_mht_tracker(
    initial_measurement: TrackingMeasurement,
    *,
    max_global_hypotheses: int,
    max_assignments_per_hypothesis: int,
    max_candidates_per_track: int,
    gate_probability: float,
    detection_probability: float,
    clutter_intensity: float,
    prune_log_weight_delta: float,
) -> MultiHypothesisTracker:
    position = np.asarray(initial_measurement.vector, dtype=float).reshape(-1)
    if position.size == 2:
        position = np.array([position[0], position[1], 0.0])
    state = np.zeros(6)
    state[:3] = position
    state_covariance = np.diag([50.0**2, 50.0**2, 50.0**2, 15.0**2, 15.0**2, 15.0**2])
    return MultiHypothesisTracker(
        initial_prior=[KalmanFilter((state, state_covariance))],
        association_param={
            "gating_probability": float(gate_probability),
            "detection_probability": float(detection_probability),
            "clutter_intensity": float(clutter_intensity),
            "max_global_hypotheses": int(max_global_hypotheses),
            "max_hypotheses_per_global_hypothesis": int(max_assignments_per_hypothesis),
            "max_measurements_per_track": int(max_candidates_per_track),
            "prune_log_weight_delta": float(prune_log_weight_delta),
        },
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )


def _predict_mht_to(
    tracker: MultiHypothesisTracker,
    *,
    current_time_s: float,
    target_time_s: float,
    acceleration_std_mps2: float,
) -> None:
    dt_s = float(target_time_s) - float(current_time_s)
    if dt_s < -1e-9:
        raise ValueError("measurements must be processed in chronological order")
    if dt_s <= 0.0:
        return
    tracker.predict_linear(
        constant_velocity_matrix(dt_s),
        white_acceleration_process_noise(dt_s, acceleration_std_mps2),
    )


def _deterministic_update_mht_hypotheses(
    tracker: MultiHypothesisTracker,
    measurement: TrackingMeasurement,
    *,
    gate_threshold: float | None,
    safety_gate_threshold: float | None,
    max_residual_norm: float | None,
    robust_update: str | None,
    inflation_alpha: float,
) -> TrackingUpdateDiagnostics:
    best_index = tracker.get_best_hypothesis_index()
    best_diagnostics: TrackingUpdateDiagnostics | None = None
    for hypothesis_index, filter_bank in enumerate(tracker.global_hypotheses):
        filter_obj = filter_bank[0]
        diagnostics = _update_filter_linear(
            filter_obj,
            measurement,
            gate_threshold=gate_threshold,
            safety_gate_threshold=safety_gate_threshold,
            max_residual_norm=max_residual_norm,
            robust_update=robust_update,
            inflation_alpha=inflation_alpha,
        )
        if hypothesis_index == best_index:
            best_diagnostics = diagnostics
    if best_diagnostics is None:
        raise RuntimeError("MHT track bank has no best hypothesis")
    return best_diagnostics


def _update_filter_linear(
    filter_obj: KalmanFilter,
    measurement: TrackingMeasurement,
    *,
    gate_threshold: float | None,
    safety_gate_threshold: float | None,
    max_residual_norm: float | None,
    robust_update: str | None,
    inflation_alpha: float,
) -> TrackingUpdateDiagnostics:
    state = np.asarray(filter_obj.get_point_estimate(), dtype=float).reshape(6)
    state_covariance = np.asarray(filter_obj.filter_state.C, dtype=float).reshape(6, 6)
    vector = np.asarray(measurement.vector, dtype=float).reshape(-1)
    covariance = np.asarray(measurement.covariance, dtype=float)
    observation = measurement_matrix(vector.size)
    plan = plan_linear_measurement_update(
        mean=state,
        covariance_matrix=state_covariance,
        measurement_vector=vector,
        measurement_covariance=covariance,
        observation_matrix=observation,
        gate_threshold=gate_threshold,
        safety_gate_threshold=safety_gate_threshold,
        max_residual_norm=max_residual_norm,
        robust_update=robust_update,
        inflation_alpha=inflation_alpha,
    )

    if plan.accepted:
        filter_obj.update_linear(plan.vector, plan.observation, plan.covariance)

    return TrackingUpdateDiagnostics(
        time_s=float(measurement.time_s),
        source=measurement.source,
        measurement_dim=plan.vector.size,
        accepted=plan.accepted,
        update_action=plan.update_action,
        nis=plan.nis,
        gate_threshold=plan.threshold,
        safety_gate_threshold=plan.safety_threshold,
        residual_gate_threshold_m=plan.residual_threshold,
        covariance_scale=plan.covariance_scale,
        inflation_alpha=float(inflation_alpha) if robust_update == "nis-inflate" else None,
        residual_norm_m=plan.residual_norm,
    )


def _selected_row_from_best_mht_assignment(
    candidates: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    tracker: MultiHypothesisTracker,
) -> pd.Series | None:
    best_index = tracker.get_best_hypothesis_index()
    history = tracker.global_hypothesis_histories[best_index]
    if not history:
        return None
    assignment = history[-1]
    if not assignment or int(assignment[0]) < 0:
        return None
    measurement_index = int(assignment[0])
    selected = candidates.iloc[measurement_index].copy()
    scored = scored_candidates.iloc[measurement_index]
    selected["association_mode"] = "track-bank"
    selected["association_action"] = "mht_assigned"
    selected["association_nis"] = float(scored["association_nis"])
    selected["association_score"] = _mht_best_negative_log_weight(tracker)
    selected["association_candidate_rows"] = int(len(candidates))
    selected["association_hypothesis_count"] = int(tracker.get_number_of_global_hypotheses())
    selected["association_best_weight"] = _mht_best_weight(tracker)
    selected["association_weight_margin"] = _mht_weight_margin(tracker)
    for column in (
        *_ASSOCIATION_COVARIANCE_COLUMNS,
        "association_covariance_mode",
        "association_cov_trace_m2",
    ):
        if column in scored:
            selected[column] = scored[column]
    return selected


def _mht_radar_measurement(
    *,
    time_s: float,
    selected: pd.Series | None,
    tracker: MultiHypothesisTracker,
) -> TrackingMeasurement:
    best_filter = tracker.get_best_hypothesis()[0]
    state = np.asarray(best_filter.get_point_estimate(), dtype=float).reshape(6)
    covariance = np.asarray(best_filter.filter_state.C, dtype=float).reshape(6, 6)
    vector = (
        state[:3]
        if selected is None
        else selected[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    )
    row_covariance = None if selected is None else _row_covariance(selected)
    return TrackingMeasurement(
        time_s=time_s,
        vector=vector,
        covariance=covariance[:3, :3] if row_covariance is None else row_covariance,
        source="radar",
    )


def _mht_radar_diagnostics(
    time_s: float,
    selected: pd.Series | None,
) -> TrackingUpdateDiagnostics:
    return TrackingUpdateDiagnostics(
        time_s=float(time_s),
        source="radar",
        measurement_dim=3,
        accepted=selected is not None,
        update_action="mht_assigned" if selected is not None else "mht_missed",
        nis=float("nan") if selected is None else float(selected["association_nis"]),
        gate_threshold=None,
        safety_gate_threshold=None,
        residual_gate_threshold_m=None,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=float("nan"),
    )


def _mht_bootstrap_diagnostics(
    measurement: TrackingMeasurement,
    *,
    gate_threshold: float | None,
    safety_gate_threshold: float | None,
    max_residual_norm: float | None,
) -> TrackingUpdateDiagnostics:
    """Return diagnostics for the sample that initialized the MHT track bank."""

    return TrackingUpdateDiagnostics(
        time_s=float(measurement.time_s),
        source=measurement.source,
        measurement_dim=measurement.vector.size,
        accepted=True,
        update_action="initialized",
        nis=0.0,
        gate_threshold=gate_threshold,
        safety_gate_threshold=safety_gate_threshold,
        residual_gate_threshold_m=max_residual_norm,
        covariance_scale=1.0,
        inflation_alpha=None,
        residual_norm_m=0.0,
    )


def _mht_record(
    measurement: TrackingMeasurement,
    tracker: MultiHypothesisTracker,
    diagnostics: TrackingUpdateDiagnostics,
    *,
    track_id: int | None = None,
    association_nis: float | None = None,
    association_score: float | None = None,
    association_mode: str | None = None,
) -> dict[str, object]:
    best_filter = tracker.get_best_hypothesis()[0]
    state = np.asarray(best_filter.get_point_estimate(), dtype=float).reshape(6)
    covariance = np.asarray(best_filter.filter_state.C, dtype=float).reshape(6, 6)
    record = {
        "time_s": float(measurement.time_s),
        "source": measurement.source,
        "state": state.copy(),
        "covariance": covariance.copy(),
        "hypothesis_count": int(tracker.get_number_of_global_hypotheses()),
        "best_hypothesis_weight": _mht_best_weight(tracker),
        "hypothesis_weight_margin": _mht_weight_margin(tracker),
        "hypotheses": _mht_hypothesis_snapshot(tracker, float(measurement.time_s)),
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


def _mht_hypothesis_snapshot(
    tracker: MultiHypothesisTracker,
    time_s: float,
) -> list[dict[str, float | int]]:
    weights = tracker.get_global_hypothesis_weights()
    order = np.argsort(-weights)
    rows: list[dict[str, float | int]] = []
    for rank, hypothesis_index in enumerate(order):
        filter_obj = tracker.global_hypotheses[int(hypothesis_index)][0]
        state = np.asarray(filter_obj.get_point_estimate(), dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(time_s),
                "rank": int(rank),
                "hypothesis_index": int(hypothesis_index),
                "weight": float(weights[int(hypothesis_index)]),
                "east_m": float(state[0]),
                "north_m": float(state[1]),
                "up_m": float(state[2]),
                "v_east_mps": float(state[3]),
                "v_north_mps": float(state[4]),
                "v_up_mps": float(state[5]),
            }
        )
    return rows


def _mht_best_weight(tracker: MultiHypothesisTracker) -> float:
    weights = tracker.get_global_hypothesis_weights()
    return float(np.max(weights)) if len(weights) else float("nan")


def _mht_weight_margin(tracker: MultiHypothesisTracker) -> float:
    weights = np.sort(tracker.get_global_hypothesis_weights())[::-1]
    if len(weights) < 2:
        return float(weights[0]) if len(weights) else float("nan")
    return float(weights[0] - weights[1])


def _mht_best_negative_log_weight(tracker: MultiHypothesisTracker) -> float:
    weight = max(_mht_best_weight(tracker), 1e-300)
    return float(-np.log(weight))


def _tracker_from_best_mht_hypothesis(
    tracker: MultiHypothesisTracker,
    current_time_s: float,
    acceleration_std_mps2: float,
) -> _MHTStateView:
    del current_time_s, acceleration_std_mps2
    return _MHTStateView(tracker.get_best_hypothesis()[0])


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


def _select_stable_radar_segments(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None,
    catprob_threshold: float | None,
    min_segment_frames: int,
    max_transition_speed_mps: float,
    rf_measurements: list[TrackingMeasurement],
    rf_score_weight: float,
    rf_time_gate_s: float,
    rf_nis_cap: float,
) -> pd.DataFrame:
    """Select stitched high-confidence Fortem track segments for sparse updates."""

    pool = _range_candidate_pool(
        radar,
        range_gate_m,
        require_fortem_range=False,
        context="stable-segments",
    )
    pool = _catprob_candidate_pool(
        pool,
        catprob_threshold,
        fallback_to_unfiltered=False,
    )
    if pool.empty or "track_id" not in pool.columns:
        return _empty_selected_radar(radar)

    segments = _stable_track_segments(
        pool,
        min_segment_frames=min_segment_frames,
        rf_measurements=rf_measurements,
        rf_score_weight=rf_score_weight,
        rf_time_gate_s=rf_time_gate_s,
        rf_nis_cap=rf_nis_cap,
    )
    if not segments:
        return _empty_selected_radar(radar)
    selected_segments = _stitch_segments(
        segments,
        max_transition_speed_mps=max_transition_speed_mps,
    )
    if not selected_segments:
        return _empty_selected_radar(radar)

    selected = pd.concat([segment.frame for segment in selected_segments], ignore_index=True)
    selected["association_mode"] = "stable-segments"
    selected["association_action"] = "stable_segment_anchor"
    selected["association_segment_count"] = int(len(selected_segments))
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _select_largest_continuous_radar_track(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None,
) -> pd.DataFrame:
    """Select the reference-paper style largest continuous Fortem track segment.

    The reference baseline describes a hard radar preselector: apply the radar
    range gate, identify continuous Fortem ``track_id`` runs, then keep the
    largest continuous track.  Unlike ``stable-segments``, this path does not
    use UAV class probability, RF-consistency scoring, or segment stitching.
    That makes it useful as a paper-reproduction diagnostic before running
    more adaptive association policies.
    """

    pool = _range_candidate_pool(
        radar,
        range_gate_m,
        require_fortem_range=range_gate_m is not None,
        context="paper-largest-continuous-track",
    )
    if pool.empty or "track_id" not in pool.columns:
        return _empty_selected_radar(radar)

    segments = _stable_track_segments(
        pool,
        min_segment_frames=1,
        rf_measurements=[],
        rf_score_weight=0.0,
        rf_time_gate_s=0.0,
        rf_nis_cap=1.0,
    )
    if not segments:
        return _empty_selected_radar(radar)

    selected_segment = max(
        segments,
        key=lambda item: (
            item.frames,
            item.end_time_s - item.start_time_s,
            item.mean_catprob,
            -item.start_time_s,
        ),
    )
    selected = selected_segment.frame.copy()
    selected["association_mode"] = "paper-largest-continuous-track"
    selected["association_action"] = "paper_largest_continuous_track_anchor"
    selected["association_segment_count"] = 1
    selected["association_segment_track_id"] = int(selected_segment.track_id)
    selected["association_segment_frames"] = int(selected_segment.frames)
    selected["association_segment_start_time_s"] = float(selected_segment.start_time_s)
    selected["association_segment_end_time_s"] = float(selected_segment.end_time_s)
    selected["association_preselector_raw_rows"] = int(len(radar))
    selected["association_preselector_range_gated_rows"] = int(len(pool))
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _select_paper_compatible_radar_track(
    radar: pd.DataFrame,
    *,
    range_gate_m: float | None,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    """Select paper-compatible anchors after hard range and class filtering."""

    range_pool = _range_candidate_pool(
        radar,
        range_gate_m,
        require_fortem_range=range_gate_m is not None,
        context="paper-compatible",
    )
    catprob_pool = _catprob_candidate_pool(
        range_pool,
        catprob_threshold,
        fallback_to_unfiltered=False,
    )
    selected = _select_largest_continuous_radar_track(catprob_pool, range_gate_m=None)
    if selected.empty:
        return selected
    selected = selected.copy()
    selected["association_mode"] = "paper-compatible"
    selected["association_action"] = "paper_compatible_largest_continuous_track_anchor"
    selected["association_preselector_raw_rows"] = int(len(radar))
    selected["association_preselector_range_gated_rows"] = int(len(range_pool))
    selected["association_preselector_track_id"] = int(selected["association_segment_track_id"].iloc[0])
    selected["association_preselector_track_rows"] = int(len(selected))
    selected["association_preselector_catprob_rows"] = int(len(catprob_pool))
    if range_gate_m is not None:
        selected["association_range_gate_m"] = float(range_gate_m)
    if catprob_threshold is not None:
        selected["association_catprob_threshold"] = float(catprob_threshold)
    return selected


def _interpolate_stable_radar_segments_to_frame_times(
    radar: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    association_mode: str,
    base_covariance: np.ndarray,
    max_gap_s: float | None,
    max_speed_mps: float | None,
    interpolated_std_scale: float,
    gap_std_mps: float,
) -> pd.DataFrame:
    """Interpolate clean stable-segment anchors onto radar frame timestamps."""

    if radar.empty or anchors.empty:
        return _empty_selected_radar(radar)
    frame_rows = _radar_frame_reference_rows(radar)
    if frame_rows.empty:
        return _empty_selected_radar(radar)

    ordered_anchors = (
        anchors.sort_values("time_s")
        .drop_duplicates(subset=["time_s"], keep="last")
        .reset_index(drop=True)
    )
    anchor_times = ordered_anchors["time_s"].to_numpy(dtype=float)
    if anchor_times.size == 0:
        return _empty_selected_radar(radar)
    frame_times = frame_rows["time_s"].to_numpy(dtype=float)
    keep = (frame_times >= anchor_times[0]) & (frame_times <= anchor_times[-1])
    outside_anchor_dropped_count = int(np.count_nonzero(~keep))
    long_gap_dropped_count = 0
    high_speed_dropped_count = 0
    if max_gap_s is not None:
        gap_keep = _within_interpolation_gap(frame_times, anchor_times, max_gap_s=float(max_gap_s))
        long_gap_dropped_count = int(np.count_nonzero(keep & ~gap_keep))
        keep &= gap_keep

    anchor_positions = ordered_anchors[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    anchor_speeds_mps = _anchor_speeds_mps(anchor_times, anchor_positions)
    if max_speed_mps is not None:
        speed_keep = _within_interpolation_speed(
            frame_times,
            anchor_times,
            anchor_positions,
            max_speed_mps=float(max_speed_mps),
        )
        high_speed_dropped_count = int(np.count_nonzero(keep & ~speed_keep))
        keep &= speed_keep

    kept_frames = frame_rows.loc[keep].reset_index(drop=True)
    if kept_frames.empty:
        return _empty_selected_radar(radar)

    anchor_by_key = {_radar_row_key(row): row for _, row in anchors.iterrows()}
    anchor_gaps_s = np.diff(anchor_times)
    max_anchor_gap_s = float(np.max(anchor_gaps_s)) if anchor_gaps_s.size else 0.0
    max_anchor_speed_mps = (
        float(np.max(anchor_speeds_mps)) if anchor_speeds_mps.size else 0.0
    )
    metadata = {
        "association_anchor_count": int(anchor_times.size),
        "association_anchor_span_s": float(anchor_times[-1] - anchor_times[0]),
        "association_max_anchor_gap_s": max_anchor_gap_s,
        "association_max_anchor_speed_mps": max_anchor_speed_mps,
        "association_interpolation_candidate_frame_count": int(len(frame_rows)),
        "association_interpolation_dropped_frame_count": int(len(frame_rows) - len(kept_frames)),
        "association_interpolation_outside_anchor_dropped_count": outside_anchor_dropped_count,
        "association_interpolation_long_gap_dropped_count": long_gap_dropped_count,
        "association_interpolation_high_speed_dropped_count": high_speed_dropped_count,
    }
    if max_gap_s is not None:
        metadata["association_interpolation_max_gap_s"] = float(max_gap_s)
    if max_speed_mps is not None:
        metadata["association_interpolation_max_speed_mps"] = float(max_speed_mps)
    interpolated_rows: list[pd.Series] = []
    modal_track_id = _modal_track_id(ordered_anchors)
    for _, frame_row in kept_frames.iterrows():
        key = _radar_row_key(frame_row)
        anchor = anchor_by_key.get(key)
        if anchor is None:
            row = _interpolated_radar_row(
                frame_row,
                anchor_times=anchor_times,
                anchor_positions=anchor_positions,
                modal_track_id=modal_track_id,
            )
            interpolation_context = _interpolation_context(time_s=float(row["time_s"]), anchor_times=anchor_times)
            interpolated_covariance = _interpolated_covariance_columns(
                base_covariance,
                std_scale=interpolated_std_scale,
                nearest_anchor_dt_s=interpolation_context["nearest_anchor_dt_s"],
                gap_std_mps=gap_std_mps,
            )
            row["association_interpolated"] = True
            row["association_action"] = "stable_segment_interpolated_anchor"
            row["association_interpolation_std_scale"] = float(interpolated_std_scale)
            row["association_interpolation_gap_std_mps"] = float(gap_std_mps)
            row["association_interpolation_gap_s"] = interpolation_context["gap_s"]
            row["association_interpolation_nearest_anchor_dt_s"] = interpolation_context[
                "nearest_anchor_dt_s"
            ]
            row["association_interpolation_gap_fraction"] = interpolation_context[
                "gap_fraction"
            ]
            for name, value in interpolated_covariance.items():
                row[name] = value
        else:
            row = anchor.copy()
            row["association_interpolated"] = False
            row["association_action"] = "stable_segment_anchor"
        row["association_mode"] = association_mode
        for name, value in metadata.items():
            row[name] = value
        interpolated_rows.append(row)

    selected = pd.DataFrame(interpolated_rows)
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in selected.columns
    ]
    return selected.sort_values(sort_columns).reset_index(drop=True)


def _radar_frame_reference_rows(radar: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for group in _radar_frame_groups(radar):
        row = group.iloc[0].copy()
        row["time_s"] = float(group["time_s"].median())
        if "frame_index" in group.columns:
            values = pd.to_numeric(group["frame_index"], errors="coerce").dropna()
            if not values.empty:
                row["frame_index"] = int(values.iloc[0])
        rows.append(row)
    if not rows:
        return radar.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def _interpolated_radar_row(
    frame_row: pd.Series,
    *,
    anchor_times: np.ndarray,
    anchor_positions: np.ndarray,
    modal_track_id: int | None,
) -> pd.Series:
    time_s = float(frame_row["time_s"])
    row = frame_row.copy()
    row["time_s"] = time_s
    for index, column in enumerate(("east_m", "north_m", "up_m")):
        row[column] = float(np.interp(time_s, anchor_times, anchor_positions[:, index]))
    if "range_m" in row.index:
        row["range_m"] = float(np.linalg.norm(row[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)))
    if modal_track_id is not None:
        row["track_id"] = modal_track_id
    return row


def _interpolation_context(
    *,
    time_s: float,
    anchor_times: np.ndarray,
) -> dict[str, float]:
    if anchor_times.size <= 1:
        return {
            "gap_s": 0.0,
            "nearest_anchor_dt_s": 0.0,
            "gap_fraction": 0.0,
        }
    insertion = int(np.searchsorted(anchor_times, float(time_s), side="left"))
    if insertion < anchor_times.size and np.isclose(anchor_times[insertion], time_s):
        return {
            "gap_s": 0.0,
            "nearest_anchor_dt_s": 0.0,
            "gap_fraction": 0.0,
        }
    right = int(np.clip(insertion, 1, anchor_times.size - 1))
    left = right - 1
    gap_s = float(anchor_times[right] - anchor_times[left])
    nearest_dt_s = min(
        abs(float(time_s) - float(anchor_times[left])),
        abs(float(anchor_times[right]) - float(time_s)),
    )
    if gap_s <= 0.0:
        fraction = 0.0
    else:
        fraction = float(nearest_dt_s / (0.5 * gap_s))
    return {
        "gap_s": gap_s,
        "nearest_anchor_dt_s": float(nearest_dt_s),
        "gap_fraction": float(np.clip(fraction, 0.0, 1.0)),
    }


def _interpolated_covariance_columns(
    covariance: np.ndarray,
    *,
    std_scale: float,
    nearest_anchor_dt_s: float,
    gap_std_mps: float,
) -> dict[str, float]:
    scaled = np.asarray(covariance, dtype=float).reshape(3, 3) * float(std_scale) ** 2
    if gap_std_mps > 0.0 and nearest_anchor_dt_s > 0.0:
        extra_variance = (float(gap_std_mps) * float(nearest_anchor_dt_s)) ** 2
        scaled = scaled + np.eye(3) * extra_variance
    return {
        "association_cov_ee": float(scaled[0, 0]),
        "association_cov_nn": float(scaled[1, 1]),
        "association_cov_uu": float(scaled[2, 2]),
        "association_cov_en": float(scaled[0, 1]),
        "association_cov_eu": float(scaled[0, 2]),
        "association_cov_nu": float(scaled[1, 2]),
        "association_covariance_mode": "stable-segment-interpolation",
        "association_cov_trace_m2": float(np.trace(scaled)),
    }


def _modal_track_id(anchors: pd.DataFrame) -> int | None:
    if "track_id" not in anchors.columns:
        return None
    values = pd.to_numeric(anchors["track_id"], errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.astype(int).mode().iloc[0])


def _within_interpolation_gap(
    frame_times: np.ndarray,
    anchor_times: np.ndarray,
    *,
    max_gap_s: float,
) -> np.ndarray:
    """Return frames bracketed by anchors no farther apart than ``max_gap_s``."""

    if frame_times.size == 0:
        return np.zeros(0, dtype=bool)
    if anchor_times.size <= 1:
        return np.isin(frame_times, anchor_times)
    insertion = np.searchsorted(anchor_times, frame_times, side="left")
    on_anchor = insertion < anchor_times.size
    on_anchor &= np.isclose(anchor_times[np.minimum(insertion, anchor_times.size - 1)], frame_times)
    right = np.clip(insertion, 1, anchor_times.size - 1)
    left = right - 1
    bracket_gap_s = anchor_times[right] - anchor_times[left]
    return on_anchor | (bracket_gap_s <= max_gap_s)


def _within_interpolation_speed(
    frame_times: np.ndarray,
    anchor_times: np.ndarray,
    anchor_positions: np.ndarray,
    *,
    max_speed_mps: float,
) -> np.ndarray:
    """Return frames bracketed by anchors no faster than ``max_speed_mps``."""

    if frame_times.size == 0:
        return np.zeros(0, dtype=bool)
    if anchor_times.size <= 1:
        return np.isin(frame_times, anchor_times)
    insertion = np.searchsorted(anchor_times, frame_times, side="left")
    on_anchor = insertion < anchor_times.size
    on_anchor &= np.isclose(anchor_times[np.minimum(insertion, anchor_times.size - 1)], frame_times)
    right = np.clip(insertion, 1, anchor_times.size - 1)
    left = right - 1
    dt_s = anchor_times[right] - anchor_times[left]
    distance_m = np.linalg.norm(anchor_positions[right] - anchor_positions[left], axis=1)
    speeds_mps = np.divide(
        distance_m,
        dt_s,
        out=np.full_like(distance_m, np.inf, dtype=float),
        where=dt_s > 0.0,
    )
    return on_anchor | (speeds_mps <= max_speed_mps)


def _anchor_speeds_mps(anchor_times: np.ndarray, anchor_positions: np.ndarray) -> np.ndarray:
    if anchor_times.size <= 1:
        return np.empty(0)
    dt_s = np.diff(anchor_times)
    distance_m = np.linalg.norm(np.diff(anchor_positions, axis=0), axis=1)
    speeds_mps = np.divide(
        distance_m,
        dt_s,
        out=np.full_like(distance_m, np.inf, dtype=float),
        where=dt_s > 0.0,
    )
    return speeds_mps[np.isfinite(speeds_mps)]


def _stable_track_segments(
    radar: pd.DataFrame,
    *,
    min_segment_frames: int,
    rf_measurements: list[TrackingMeasurement],
    rf_score_weight: float,
    rf_time_gate_s: float,
    rf_nis_cap: float,
) -> list[_TrackSegment]:
    segments: list[_TrackSegment] = []
    for track_id, track_rows in radar.groupby("track_id", sort=True):
        ordered = track_rows.sort_values(
            ["frame_index" if "frame_index" in track_rows.columns else "time_s", "time_s"]
        )
        frame_values = (
            pd.to_numeric(ordered["frame_index"], errors="coerce").to_numpy(dtype=float)
            if "frame_index" in ordered.columns
            else ordered["time_s"].to_numpy(dtype=float)
        )
        splits = np.r_[
            0,
            np.where(np.diff(frame_values) > _segment_gap_threshold(frame_values))[0] + 1,
            len(ordered),
        ]
        for start, end in zip(splits[:-1], splits[1:]):
            frame = ordered.iloc[int(start) : int(end)].copy()
            if len(frame) < int(min_segment_frames):
                continue
            positions = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
            times = frame["time_s"].to_numpy(dtype=float)
            catprob = (
                pd.to_numeric(frame["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
                if "cat_prob_uav" in frame.columns
                else np.ones(len(frame), dtype=float)
            )
            mean_catprob = float(np.nanmean(catprob))
            if not np.isfinite(mean_catprob):
                mean_catprob = 0.0
            (
                rf_support_count,
                rf_mean_nis,
                rf_score_adjustment,
            ) = _stable_segment_rf_consistency(
                frame,
                rf_measurements=rf_measurements,
                rf_score_weight=rf_score_weight,
                rf_time_gate_s=rf_time_gate_s,
                rf_nis_cap=rf_nis_cap,
            )
            frame["association_segment_rf_support_count"] = int(rf_support_count)
            frame["association_segment_rf_score_adjustment"] = float(rf_score_adjustment)
            frame["association_segment_base_score"] = float(len(frame)) * max(
                mean_catprob,
                0.0,
            )
            frame["association_segment_score"] = frame["association_segment_base_score"] + float(
                rf_score_adjustment
            )
            if rf_mean_nis is not None:
                frame["association_segment_rf_mean_nis"] = float(rf_mean_nis)
            segments.append(
                _TrackSegment(
                    frame=frame,
                    track_id=int(track_id),
                    start_time_s=float(times[0]),
                    end_time_s=float(times[-1]),
                    start_position_m=positions[0],
                    end_position_m=positions[-1],
                    frames=int(len(frame)),
                    mean_catprob=mean_catprob,
                    rf_support_count=rf_support_count,
                    rf_mean_nis=rf_mean_nis,
                    rf_score_adjustment=rf_score_adjustment,
                )
            )
    return sorted(segments, key=lambda item: (item.start_time_s, -item.score))


def _stable_segment_rf_consistency(
    frame: pd.DataFrame,
    *,
    rf_measurements: list[TrackingMeasurement],
    rf_score_weight: float,
    rf_time_gate_s: float,
    rf_nis_cap: float,
) -> tuple[int, float | None, float]:
    """Return RF support count, mean horizontal RF NIS, and score adjustment."""

    if frame.empty or rf_score_weight <= 0.0 or rf_time_gate_s < 0.0 or not rf_measurements:
        return 0, None, 0.0
    times = frame["time_s"].to_numpy(dtype=float)
    positions_xy = frame[["east_m", "north_m"]].to_numpy(dtype=float)
    if times.size == 0 or not np.isfinite(times).any():
        return 0, None, 0.0

    nises: list[float] = []
    for measurement in rf_measurements:
        if measurement.vector.size < 2 or measurement.covariance.shape[0] < 2:
            continue
        time_s = float(measurement.time_s)
        dt_to_segment_s = _time_distance_to_interval(
            time_s,
            start_s=float(times[0]),
            end_s=float(times[-1]),
        )
        if dt_to_segment_s > float(rf_time_gate_s):
            continue
        interpolated_xy = np.array(
            [
                np.interp(time_s, times, positions_xy[:, 0]),
                np.interp(time_s, times, positions_xy[:, 1]),
            ],
            dtype=float,
        )
        residual = interpolated_xy - np.asarray(measurement.vector[:2], dtype=float)
        covariance = np.asarray(measurement.covariance[:2, :2], dtype=float)
        if not np.isfinite(covariance).all() or not np.isfinite(residual).all():
            continue
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance)
        nis = float(residual.T @ precision @ residual)
        if np.isfinite(nis):
            nises.append(float(min(nis, rf_nis_cap)))
    if not nises:
        return 0, None, 0.0
    mean_nis = float(np.mean(nises))
    adjustment = -float(rf_score_weight) * float(len(nises)) * mean_nis
    return int(len(nises)), mean_nis, adjustment


def _time_distance_to_interval(time_s: float, *, start_s: float, end_s: float) -> float:
    if time_s < start_s:
        return float(start_s - time_s)
    if time_s > end_s:
        return float(time_s - end_s)
    return 0.0


def _segment_gap_threshold(frame_values: np.ndarray) -> float:
    values = np.sort(np.asarray(frame_values, dtype=float).reshape(-1))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("inf")
    diffs = np.diff(values)
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return float("inf")
    if _integer_like(values):
        return 1.5
    return 1.5 * float(np.median(positive))


def _stitch_segments(
    segments: list[_TrackSegment],
    *,
    max_transition_speed_mps: float,
) -> list[_TrackSegment]:
    ordered = sorted(segments, key=lambda item: (item.start_time_s, item.end_time_s))
    best_paths: list[list[_TrackSegment]] = []
    best_scores: list[float] = []
    for segment in ordered:
        best_path = [segment]
        best_score = segment.score
        for index, previous in enumerate(ordered[: len(best_paths)]):
            if not _segments_can_follow(
                previous,
                segment,
                max_transition_speed_mps=max_transition_speed_mps,
            ):
                continue
            score = best_scores[index] + segment.score
            if score > best_score:
                best_score = score
                best_path = [*best_paths[index], segment]
        best_paths.append(best_path)
        best_scores.append(best_score)
    if not best_paths:
        return []
    return best_paths[int(np.argmax(best_scores))]


def _segments_can_follow(
    previous: _TrackSegment,
    current: _TrackSegment,
    *,
    max_transition_speed_mps: float,
) -> bool:
    if current.start_time_s <= previous.end_time_s:
        return False
    dt_s = current.start_time_s - previous.end_time_s
    if dt_s <= 0.0:
        return False
    distance_m = float(np.linalg.norm(current.start_position_m - previous.end_position_m))
    return distance_m / dt_s <= float(max_transition_speed_mps)


def _range_candidate_pool(
    candidates: pd.DataFrame,
    range_gate_m: float | None,
    *,
    require_fortem_range: bool = False,
    context: str = "radar-association",
) -> pd.DataFrame:
    if candidates.empty or range_gate_m is None:
        return candidates.copy()
    ranges, range_source = _candidate_ranges_m(
        candidates,
        require_fortem_range=require_fortem_range,
        context=context,
    )
    pool = candidates.loc[np.isfinite(ranges) & (ranges <= float(range_gate_m))].copy()
    if not pool.empty:
        pool["association_range_gate_m"] = float(range_gate_m)
        pool["association_range_source"] = range_source
        pool["association_range_fortem_required"] = bool(require_fortem_range)
    return pool


def _candidate_ranges_m(
    candidates: pd.DataFrame,
    *,
    require_fortem_range: bool = False,
    context: str = "radar-association",
) -> tuple[np.ndarray, str]:
    if "range_m" in candidates.columns:
        ranges = pd.to_numeric(candidates["range_m"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(ranges)
        if require_fortem_range and not finite.all():
            finite_fraction = float(np.mean(finite)) if ranges.size else 0.0
            raise ValueError(
                f"{context} range gating requires finite Fortem range_m for all rows; "
                f"observed finite fraction {finite_fraction:.3f}"
            )
        if finite.any():
            return ranges, "range_m"
        if require_fortem_range:
            raise ValueError(f"{context} range gating found no finite Fortem range_m values")
    elif require_fortem_range:
        raise ValueError(
            f"{context} range gating requires Fortem range_m; "
            "falling back to ENU norm would change the 800 m paper gate"
        )
    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    return np.linalg.norm(positions, axis=1), "enu_norm_fallback"


def _radar_event_key(event: dict[str, object]) -> object:
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if "frame_index" in candidates.columns:
        values = pd.to_numeric(candidates["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return ("frame_index", int(values.iloc[0]))
    time_s = event.get("time_s")
    if time_s is None:
        time_s = float(candidates["time_s"].median())
    return ("time_s", round(float(time_s), 9))


def _radar_row_key(row: pd.Series) -> object:
    if "frame_index" in row.index and np.isfinite(float(row["frame_index"])):
        return ("frame_index", int(row["frame_index"]))
    return ("time_s", round(float(row["time_s"]), 9))


def _integer_like(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size and np.allclose(finite, np.round(finite)))


def _initial_measurement_and_row(
    event: dict[str, object],
    *,
    association: str,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None = None,
    stable_anchor_by_key: dict[object, pd.Series] | None = None,
    candidate_catprob_threshold: float | None = None,
    truth: pd.DataFrame | None,
    truth_gate_m: float,
    truth_time_gate_s: float,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> tuple[TrackingMeasurement, pd.Series | None] | None:
    """Return the bootstrap measurement and selected radar row, if any."""

    if event["kind"] == "rf":
        measurement = event["measurement"]
        assert isinstance(measurement, TrackingMeasurement)
        return measurement, None
    candidates = event["candidates"]
    assert isinstance(candidates, pd.DataFrame)
    if association == "oracle-nearest-truth":
        selected = _oracle_nearest_truth(
            candidates,
            truth=truth,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
    elif association in _STABLE_SEGMENT_ASSOCIATION_MODES:
        selected = (
            None
            if stable_anchor_by_key is None
            else stable_anchor_by_key.get(_radar_event_key(event))
        )
    else:
        candidates = _catprob_candidate_pool(
            candidates,
            candidate_catprob_threshold,
            fallback_to_unfiltered=False,
        )
        if candidates.empty:
            return None
        selected = _highest_catprob(candidates)
    if selected is None:
        return None
    selected = selected.copy()
    selected = _annotate_radar_geometry_covariance(selected, covariance, covariance_config)
    if radar_covariance_fn is not None:
        covariance = radar_covariance_fn(selected, covariance)
    return _radar_row_to_measurement(selected, covariance), selected


def _initial_measurement(
    event: dict[str, object],
    *,
    association: str,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None = None,
    stable_anchor_by_key: dict[object, pd.Series] | None = None,
    candidate_catprob_threshold: float | None = None,
    truth: pd.DataFrame | None,
    truth_gate_m: float,
    truth_time_gate_s: float,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> TrackingMeasurement | None:
    initial = _initial_measurement_and_row(
        event,
        association=association,
        covariance=covariance,
        covariance_config=covariance_config,
        stable_anchor_by_key=stable_anchor_by_key,
        candidate_catprob_threshold=candidate_catprob_threshold,
        truth=truth,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
        radar_covariance_fn=radar_covariance_fn,
    )
    return None if initial is None else initial[0]


def _select_radar_candidate(
    candidates: pd.DataFrame,
    *,
    association: str,
    tracker: AsyncConstantVelocityKalmanTracker,
    covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None = None,
    truth: pd.DataFrame | None,
    current_track_id: int | None,
    track_switch_nis_ratio: float,
    candidate_catprob_threshold: float | None,
    geometry_velocity_std_mps: float,
    geometry_velocity_weight: float,
    geometry_switch_penalty: float,
    geometry_catprob_weight: float,
    rf_measurements: list[TrackingMeasurement] | None = None,
    rf_anchor_weight: float = 0.35,
    rf_anchor_time_gate_s: float = 2.0,
    rf_anchor_nis_cap: float = 25.0,
    rf_anchor_gate_nis: float = 25.0,
    pda_nis_temperature: float,
    pda_catprob_exponent: float,
    stable_anchor_by_key: dict[object, pd.Series] | None = None,
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
    if association in _STABLE_SEGMENT_ASSOCIATION_MODES:
        if stable_anchor_by_key is None:
            return None
        selected = stable_anchor_by_key.get(_radar_event_key({"candidates": candidates}))
        if selected is None:
            return None
        selected = selected.copy()
        selected["association_mode"] = association
        if association == "paper-largest-continuous-track":
            selected["association_action"] = "paper_largest_continuous_track_update"
        elif bool(selected.get("association_interpolated", False)):
            selected["association_action"] = "stable_segment_interpolated_update"
        else:
            selected["association_action"] = "stable_segment_update"
        selected["association_candidate_rows"] = int(len(candidates))
        return _annotate_radar_geometry_covariance(selected, covariance, covariance_config)

    candidates = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    if candidates.empty:
        return None
    if association == "stable-segments-hybrid" and stable_anchor_by_key is not None:
        selected = stable_anchor_by_key.get(_radar_event_key({"candidates": candidates}))
        if selected is not None:
            selected = selected.copy()
            selected["association_mode"] = association
            selected["association_action"] = "stable_segment_hybrid_update"
            selected["association_candidate_rows"] = int(len(candidates))
            return _annotate_radar_geometry_covariance(selected, covariance, covariance_config)
    scored = _nis_scored_candidates(
        candidates,
        tracker,
        covariance,
        covariance_config=covariance_config,
    )
    if scored.empty:
        return None
    if association == "stable-segments-hybrid":
        best = scored.loc[scored["association_nis"].idxmin()].copy()
        best["association_mode"] = association
        best["association_action"] = "stable_segment_hybrid_prediction_nis"
        return best
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
    if association == "rf-anchored-nis":
        rf_scored = _rf_anchor_scored_candidates(
            scored,
            rf_measurements=rf_measurements,
            anchor_weight=rf_anchor_weight,
            time_gate_s=rf_anchor_time_gate_s,
            nis_cap=rf_anchor_nis_cap,
        )
        best = rf_scored.loc[rf_scored["association_score"].idxmin()].copy()
        best["association_action"] = "rf_anchored_nis"
        return best
    if association == "rf-gated-nis":
        rf_scored = _rf_anchor_scored_candidates(
            scored,
            rf_measurements=rf_measurements,
            anchor_weight=rf_anchor_weight,
            time_gate_s=rf_anchor_time_gate_s,
            nis_cap=rf_anchor_nis_cap,
        )
        gated = _rf_anchor_gate_candidates(rf_scored, gate_nis=rf_anchor_gate_nis)
        if gated.empty:
            return None
        best = gated.loc[gated["association_score"].idxmin()].copy()
        best["association_action"] = "rf_gated_nis"
        return best
    if association == "pda-mixture":
        return _pda_mixture_candidate(
            scored,
            base_covariance=covariance,
            nis_temperature=pda_nis_temperature,
            catprob_exponent=pda_catprob_exponent,
        )
    score_column = "association_score" if "association_score" in scored.columns else "association_nis"
    best = scored.loc[scored[score_column].idxmin()].copy()
    if association == "prediction-nis" or current_track_id is None:
        return best
    if association != "track-continuity":
        raise ValueError(f"unknown radar association mode {association!r}")

    if "track_id" not in scored.columns:
        return best
    current = scored.loc[
        scored["track_id"].map(lambda value: _track_id_equals(value, current_track_id))
    ]
    if current.empty:
        return best
    current_best = current.loc[current[score_column].idxmin()].copy()
    if _optional_track_id(best) == current_track_id:
        return best
    if float(best[score_column]) < float(current_best[score_column]) * float(
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
    *,
    fallback_to_unfiltered: bool = True,
) -> pd.DataFrame:
    if candidate_catprob_threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates
    catprob = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    threshold = float(candidate_catprob_threshold)
    if _env_flag(_SOFT_CATPROB_RETENTION_ENV):
        pool = candidates.copy()
        values = catprob.fillna(0.0).to_numpy(dtype=float)
        below = values < threshold
        scale = max(threshold, 1.0e-9)
        penalty = _env_float(_SOFT_CATPROB_BELOW_THRESHOLD_PENALTY_ENV, 3.0)
        pool["association_catprob_threshold"] = threshold
        pool["association_catprob_fallback"] = below
        pool["association_catprob_below_threshold"] = below
        pool["association_catprob_penalty"] = penalty * np.square(
            np.maximum(0.0, threshold - values) / scale
        )
        pool["association_catprob_retention_mode"] = "soft"
        pool["association_catprob_candidate_rows"] = int(len(candidates))
        return pool
    keep = catprob >= threshold
    fallback = bool(fallback_to_unfiltered and not candidates.empty and not bool(keep.any()))
    if fallback:
        # A hard class-probability cut must not turn the whole radar frame into
        # a missed detection. When no candidate reaches the threshold, retain
        # the unfiltered frame and let downstream geometry/temporal scoring
        # decide, while recording that the catProb gate was bypassed.
        pool = candidates.copy()
    else:
        pool = candidates.loc[keep].copy()
    pool["association_catprob_threshold"] = threshold
    pool["association_catprob_fallback"] = fallback
    pool["association_catprob_candidate_rows"] = int(len(candidates))
    if fallback:
        pool["association_catprob_fallback_reason"] = "all_candidates_below_threshold"
    return pool


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
    *,
    covariance_config: _RadarGeometryCovarianceConfig | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.iloc[0:0].copy()
    observation = measurement_matrix(3)
    state_position = observation @ tracker.state
    predicted_covariance = observation @ tracker.covariance_matrix @ observation.T
    vectors = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = vectors - state_position
    measurement_covariances = _candidate_covariances(candidates, covariance, covariance_config)

    association_nis = np.empty(len(candidates), dtype=float)
    for index, (residual, measurement_covariance) in enumerate(
        zip(residuals, measurement_covariances, strict=True)
    ):
        innovation_covariance = predicted_covariance + measurement_covariance
        try:
            precision = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(innovation_covariance)
        association_nis[index] = float(residual.T @ precision @ residual)

    scored = candidates.copy()
    scored["association_nis"] = association_nis
    scored["association_candidate_rows"] = int(len(candidates))
    if covariance_config is not None and covariance_config.enabled:
        scored = _attach_covariance_columns(
            scored,
            measurement_covariances,
            covariance_mode="radar-geometry",
        )
    if "association_catprob_penalty" in scored.columns:
        scored["association_score"] = scored["association_nis"].to_numpy(dtype=float) + pd.to_numeric(
            scored["association_catprob_penalty"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
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


def _rf_anchor_scored_candidates(
    candidates: pd.DataFrame,
    *,
    rf_measurements: list[TrackingMeasurement] | None,
    anchor_weight: float,
    time_gate_s: float,
    nis_cap: float,
) -> pd.DataFrame:
    scored = candidates.copy()
    scored["association_score"] = scored["association_nis"].to_numpy(dtype=float)
    scored["association_anchor_penalty"] = 0.0
    if anchor_weight <= 0.0 or not rf_measurements:
        return scored

    time_s = float(candidates["time_s"].median())
    anchor = _latest_rf_anchor(
        rf_measurements,
        time_s=time_s,
        time_gate_s=time_gate_s,
    )
    if anchor is None:
        return scored
    if anchor.vector.size < 2 or anchor.covariance.shape[0] < 2:
        return scored

    covariance = np.asarray(anchor.covariance[:2, :2], dtype=float)
    if not np.isfinite(covariance).all():
        return scored
    try:
        precision = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(covariance)

    vectors = candidates[["east_m", "north_m"]].to_numpy(dtype=float)
    residuals = vectors - np.asarray(anchor.vector[:2], dtype=float)
    anchor_nis = np.einsum("ij,jk,ik->i", residuals, precision, residuals)
    anchor_nis = np.where(np.isfinite(anchor_nis), anchor_nis, np.inf)
    capped_nis = np.minimum(anchor_nis, float(nis_cap))
    penalty = float(anchor_weight) * capped_nis
    scored["association_anchor_nis"] = anchor_nis
    scored["association_anchor_penalty"] = penalty
    scored["association_anchor_time_delta_s"] = float(time_s - anchor.time_s)
    scored["association_anchor_weight"] = float(anchor_weight)
    scored["association_anchor_nis_cap"] = float(nis_cap)
    scored["association_score"] = scored["association_score"].to_numpy(dtype=float) + penalty
    return scored


def _rf_anchor_gate_candidates(candidates: pd.DataFrame, *, gate_nis: float) -> pd.DataFrame:
    """Keep candidates consistent with the latest RF anchor when one is available."""

    gated = candidates.copy()
    gated["association_anchor_gate_nis"] = float(gate_nis)
    if "association_anchor_nis" not in gated.columns:
        gated["association_anchor_gate_rejected_count"] = 0
        gated["association_anchor_gate_candidate_count"] = int(len(gated))
        return gated

    anchor_nis = pd.to_numeric(gated["association_anchor_nis"], errors="coerce")
    keep = anchor_nis <= float(gate_nis)
    rejected_count = int((~keep).sum())
    kept = gated.loc[keep].copy()
    if kept.empty:
        return kept
    kept["association_anchor_gate_rejected_count"] = rejected_count
    kept["association_anchor_gate_candidate_count"] = int(len(gated))
    return kept


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
    measurement_covariance = np.einsum(
        "i,ijk->jk",
        weights,
        _candidate_row_covariances(candidates, base_covariance),
    )
    covariance = measurement_covariance + spread

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
    selected["association_covariance_mode"] = _pda_covariance_mode(candidates)
    selected["association_cov_trace_m2"] = float(np.trace(covariance))
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



def _candidate_covariances(
    candidates: pd.DataFrame,
    base_covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None,
) -> np.ndarray:
    base = np.asarray(base_covariance, dtype=float).reshape(3, 3)
    if len(candidates) == 0:
        return np.empty((0, 3, 3), dtype=float)
    covariances = []
    for _, row in candidates.iterrows():
        row_covariance = _row_covariance(row)
        if row_covariance is not None:
            covariances.append(row_covariance)
        elif covariance_config is not None and covariance_config.enabled:
            covariances.append(_radar_geometry_covariance_for_row(row, base, covariance_config))
        else:
            covariances.append(base)
    return np.stack(covariances, axis=0)


def _candidate_covariance_tensor(
    candidates: pd.DataFrame,
    base_covariance: np.ndarray,
) -> np.ndarray:
    covariances = _candidate_row_covariances(candidates, base_covariance)
    return np.moveaxis(covariances, 0, -1)


def _candidate_row_covariances(
    candidates: pd.DataFrame,
    base_covariance: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_covariance, dtype=float).reshape(3, 3)
    if len(candidates) == 0:
        return np.empty((0, 3, 3), dtype=float)
    covariances = []
    for _, row in candidates.iterrows():
        row_covariance = _row_covariance(row)
        covariances.append(base if row_covariance is None else row_covariance)
    return np.stack(covariances, axis=0)


def _annotate_radar_geometry_covariance(
    row: pd.Series,
    base_covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig | None,
) -> pd.Series:
    if covariance_config is None or not covariance_config.enabled:
        return row
    if _row_covariance(row) is not None:
        return row
    covariance = _radar_geometry_covariance_for_row(row, base_covariance, covariance_config)
    return _attach_covariance_columns(
        row.to_frame().T,
        covariance[None, :, :],
        covariance_mode="radar-geometry",
    ).iloc[0]


def _attach_covariance_columns(
    frame: pd.DataFrame,
    covariances: np.ndarray,
    *,
    covariance_mode: str,
) -> pd.DataFrame:
    annotated = frame.copy()
    covariances = np.asarray(covariances, dtype=float)
    annotated["association_cov_ee"] = covariances[:, 0, 0]
    annotated["association_cov_nn"] = covariances[:, 1, 1]
    annotated["association_cov_uu"] = covariances[:, 2, 2]
    annotated["association_cov_en"] = covariances[:, 0, 1]
    annotated["association_cov_eu"] = covariances[:, 0, 2]
    annotated["association_cov_nu"] = covariances[:, 1, 2]
    annotated["association_covariance_mode"] = covariance_mode
    annotated["association_cov_trace_m2"] = np.trace(covariances, axis1=1, axis2=2)
    return annotated


def _radar_geometry_covariance_for_row(
    row: pd.Series,
    base_covariance: np.ndarray,
    covariance_config: _RadarGeometryCovarianceConfig,
) -> np.ndarray:
    base = np.asarray(base_covariance, dtype=float).reshape(3, 3)
    try:
        position = row[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError):
        return base
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return base
    position_norm = float(np.linalg.norm(position))
    range_m = _radar_row_range_m(row, fallback=position_norm)
    if position_norm <= 1.0e-9 or range_m <= 0.0:
        return base

    radial_std_m = max(
        float(covariance_config.radial_std_m),
        float(covariance_config.radial_range_fraction) * range_m,
        float(covariance_config.min_std_m),
    )
    crossrange_std_m = float(
        np.clip(
            covariance_config.crossrange_angle_std_rad * range_m,
            covariance_config.crossrange_min_std_m,
            covariance_config.crossrange_max_std_m,
        )
    )
    unit_los = position / position_norm
    los_projector = np.outer(unit_los, unit_los)
    covariance = (crossrange_std_m**2) * np.eye(3)
    covariance += (radial_std_m**2 - crossrange_std_m**2) * los_projector
    covariance += (float(covariance_config.min_std_m) ** 2) * np.eye(3)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        return base
    return covariance


def _radar_row_range_m(row: pd.Series, *, fallback: float) -> float:
    for column in ("range_m", "slant_range_m", "radar_range_m"):
        if column in row.index:
            value = _optional_float(row[column])
            if value is not None and np.isfinite(value) and value > 0.0:
                return float(value)
    return float(fallback)


def _pda_covariance_mode(candidates: pd.DataFrame) -> str:
    if "association_covariance_mode" not in candidates.columns:
        return "pda-mixture"
    modes = [
        str(mode)
        for mode in candidates["association_covariance_mode"].dropna().unique().tolist()
        if str(mode)
    ]
    if not modes:
        return "pda-mixture"
    if len(modes) == 1:
        return f"pda-mixture+{modes[0]}"
    return "pda-mixture+row-covariance"

def _radar_row_to_measurement(row: pd.Series, covariance: np.ndarray) -> TrackingMeasurement:
    row_covariance = _row_covariance(row)
    position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
    measurement_covariance = covariance if row_covariance is None else row_covariance
    velocity = _row_velocity_vector(row) if _env_flag(_RADAR_VELOCITY_UPDATE_ENV) else None
    if velocity is not None:
        measurement_covariance = _position_velocity_covariance(
            measurement_covariance,
            velocity_std_mps=_env_float(_RADAR_VELOCITY_STD_MPS_ENV, 12.0),
        )
        vector = np.concatenate([position, velocity])
    else:
        vector = position
    return TrackingMeasurement(
        time_s=float(row["time_s"]),
        vector=vector,
        covariance=measurement_covariance,
        source="radar",
    )


def _row_covariance(row: pd.Series) -> np.ndarray | None:
    """Return row-wise radar position covariance, if available.

    Association-specific covariance columns remain preferred, but calibrated
    heteroscedastic ``cov_*`` columns are accepted as a second choice.
    """

    for prefix in ("association_cov", "cov"):
        columns = tuple(f"{prefix}_{suffix}" for suffix in ("ee", "nn", "uu", "en", "eu", "nu"))
        if not all(column in row for column in columns):
            continue
        values = [float(row[column]) for column in columns]
        if not np.isfinite(values).all():
            continue
        ee, nn, uu, en, eu, nu = values
        covariance = np.array(
            [
                [ee, en, eu],
                [en, nn, nu],
                [eu, nu, uu],
            ],
            dtype=float,
        )
        covariance = 0.5 * (covariance + covariance.T)
        if np.all(np.diag(covariance) > 0.0):
            return covariance
    return None


def _frame_has_row_covariance(frame: pd.DataFrame) -> bool:
    return any(
        all(f"{prefix}_{suffix}" in frame.columns for suffix in ("ee", "nn", "uu", "en", "eu", "nu"))
        for prefix in ("association_cov", "cov")
    )


def _row_velocity_vector(row: pd.Series) -> np.ndarray | None:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in row.index for column in required):
        return None
    try:
        velocity = np.array(
            [
                float(row["velocity_east_mps"]),
                float(row["velocity_north_mps"]),
                -float(row["velocity_down_mps"]),
            ],
            dtype=float,
        )
    except (TypeError, ValueError):
        return None
    return velocity if np.isfinite(velocity).all() else None


def _position_velocity_covariance(
    position_covariance: np.ndarray,
    *,
    velocity_std_mps: float,
) -> np.ndarray:
    velocity_variance = float(velocity_std_mps) ** 2
    covariance = np.zeros((6, 6), dtype=float)
    covariance[:3, :3] = np.asarray(position_covariance, dtype=float).reshape(3, 3)
    covariance[3:, 3:] = np.eye(3) * velocity_variance
    return covariance


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


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


def _max_residual_norm_for_measurement(
    measurement: TrackingMeasurement,
    *,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
) -> float | None:
    return max_residual_norm_for_measurement(
        measurement,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )


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


def _selected_row_record_kwargs(
    row: pd.Series | None,
    association: str,
) -> dict[str, int | float | str | None]:
    """Return optional record metadata derived from a selected radar row."""

    if row is None:
        return {}
    mode = row.get("association_mode", association)
    if mode is None or pd.isna(mode):
        mode = association
    return {
        "track_id": _optional_track_id(row),
        "association_nis": _optional_float(row.get("association_nis")),
        "association_score": _optional_float(row.get("association_score")),
        "association_mode": str(mode),
    }


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
        "association_action",
        "association_nis",
        "association_score",
        "association_replay_accepted",
        "association_replay_nis",
        "association_replay_update_action",
        "association_candidate_rows",
        "association_anchor_nis",
        "association_anchor_penalty",
        "association_anchor_time_delta_s",
        "association_anchor_weight",
        "association_anchor_nis_cap",
        "association_anchor_gate_nis",
        "association_anchor_gate_rejected_count",
        "association_anchor_gate_candidate_count",
        "association_nis_gate_threshold",
        "association_nis_gate_rejected_count",
        "association_preselector_raw_rows",
        "association_preselector_track_id",
        "association_preselector_track_rows",
        "association_preselector_catprob_rows",
        "association_preselector_range_gated_rows",
        "association_effective_candidates",
        "association_weight_max",
        "association_position_spread_trace_m2",
        "association_segment_base_score",
        "association_segment_score",
        "association_segment_rf_support_count",
        "association_segment_rf_mean_nis",
        "association_segment_rf_score_adjustment",
        "association_interpolated",
        "association_anchor_count",
        "association_interpolation_dropped_frame_count",
        "association_interpolation_std_scale",
        "association_interpolation_gap_std_mps",
        "association_interpolation_gap_s",
        "association_interpolation_nearest_anchor_dt_s",
        "association_interpolation_gap_fraction",
        "association_cov_ee",
        "association_cov_nn",
        "association_cov_uu",
        "association_cov_en",
        "association_cov_eu",
        "association_cov_nu",
        "association_covariance_mode",
        "association_cov_trace_m2",
    ):
        selected[column] = []
    return selected


def _track_id_equals(value: object, track_id: int) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(numeric):
        return False
    return int(numeric) == int(track_id)


def _optional_track_id(row: pd.Series) -> int | None:
    return optional_int(row.get("track_id"))


def _optional_float(value: object) -> float | None:
    return optional_float(value)
