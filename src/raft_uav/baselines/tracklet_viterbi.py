"""Truth-free sequence-level radar tracklet association.

This module selects one coherent radar path over all radar frames and then
replays that path through the existing asynchronous CV Kalman baseline.  The
objective combines RF-anchor consistency, Fortem track-ID continuity, CV motion
feasibility, radar velocity consistency, class probability, range, and a
missed-detection branch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import os

import numpy as np
import pandas as pd
from pyrecest.filters.sequence_association import (
    SequenceAssociationNode,
    SequenceTransitionContext,
    solve_top_k_viterbi_sequence_associations,
)

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)
from raft_uav.baselines.learned_radar_likelihood import (
    LearnedRadarAssociationModel,
    radar_association_feature_frame,
)
from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_track_id

RadarCovarianceFn = Callable[[pd.Series, np.ndarray], np.ndarray]
_DO_NO_HARM_RADAR_POLICY_ENV = "RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY"
_TRACKLET_SOFT_TOP_K_PATHS_ENV = "RAFT_UAV_TRACKLET_SOFT_TOP_K_PATHS"
_TRACKLET_SOFT_PATH_TEMPERATURE_ENV = "RAFT_UAV_TRACKLET_SOFT_PATH_TEMPERATURE"


@dataclass(frozen=True)
class TrackletViterbiAssociationConfig:
    """Configuration for sequence-level radar association."""

    max_candidates_per_frame: int = 8
    missed_detection_cost: float = 7.0
    consecutive_miss_cost: float = 1.0
    track_switch_cost: float = 8.0
    missing_track_id_cost: float = 1.0
    catprob_weight: float = 2.5
    anchor_nis_weight: float = 0.35
    transition_nis_weight: float = 1.0
    velocity_nis_weight: float = 0.15
    transition_position_std_m: float = 40.0
    transition_speed_std_mps: float = 18.0
    velocity_std_mps: float = 12.0
    max_speed_mps: float = 55.0
    max_speed_penalty: float = 10.0
    range_gate_m: float | None = 850.0
    range_gate_slack_m: float = 150.0
    range_penalty: float = 10.0
    reacquisition_miss_streak_threshold: int = 2
    reacquisition_gate_nis: float = 25.0
    reacquisition_gate_growth: float = 0.75
    reacquisition_reward: float = 3.0
    reacquisition_outside_gate_penalty: float = 4.0
    use_rf_anchor: bool = True
    learned_candidate_model: LearnedRadarAssociationModel | None = None
    learned_candidate_score_mode: str = "additive"
    min_learned_candidate_probability: float = 1.0e-9
    min_catprob: float = 1.0e-3
    soft_top_k_paths: int = 1
    soft_path_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.max_candidates_per_frame < 1:
            raise ValueError("max_candidates_per_frame must be positive")
        if self.soft_top_k_paths < 1:
            raise ValueError("soft_top_k_paths must be positive")
        if self.soft_path_temperature <= 0.0:
            raise ValueError("soft_path_temperature must be positive")
        if self.reacquisition_miss_streak_threshold < 1:
            raise ValueError("reacquisition_miss_streak_threshold must be positive")
        positive = (
            "transition_position_std_m",
            "transition_speed_std_mps",
            "velocity_std_mps",
            "max_speed_mps",
            "reacquisition_gate_nis",
        )
        nonnegative = (
            "missed_detection_cost",
            "consecutive_miss_cost",
            "track_switch_cost",
            "missing_track_id_cost",
            "catprob_weight",
            "anchor_nis_weight",
            "transition_nis_weight",
            "velocity_nis_weight",
            "max_speed_penalty",
            "range_gate_slack_m",
            "range_penalty",
            "reacquisition_gate_growth",
            "reacquisition_reward",
            "reacquisition_outside_gate_penalty",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.range_gate_m is not None and float(self.range_gate_m) <= 0.0:
            raise ValueError("range_gate_m must be positive or None")
        if not 0.0 < float(self.min_catprob) <= 1.0:
            raise ValueError("min_catprob must be in (0, 1]")
        if not 0.0 < float(self.min_learned_candidate_probability) <= 1.0:
            raise ValueError("min_learned_candidate_probability must be in (0, 1]")
        mode = str(self.learned_candidate_score_mode).strip().lower()
        if mode not in {"additive", "replace"}:
            raise ValueError("learned_candidate_score_mode must be 'additive' or 'replace'")


@dataclass(frozen=True)
class _AnchorState:
    state: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class _ViterbiNode:
    event_index: int
    event_key: tuple[str, int | float]
    time_s: float
    row: pd.Series | None
    position: np.ndarray | None
    velocity: np.ndarray | None
    track_id: int | None
    unary_cost: float
    anchor_nis: float
    catprob_cost: float
    range_cost: float
    is_miss: bool = False
    has_anchor: bool = False
    learned_cost: float = 0.0
    learned_probability: float | None = None
    base_unary_cost: float | None = None


def run_async_cv_baseline_with_tracklet_viterbi_association(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
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
    candidate_catprob_threshold: float | None = 0.4,
    config: TrackletViterbiAssociationConfig | None = None,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Run CV fusion after Viterbi radar-tracklet selection."""

    from raft_uav.baselines.radar_association import (
        _empty_selected_radar,
        _events,
        _initial_measurement,
        _selected_rows_frame,
    )

    cfg = config or TrackletViterbiAssociationConfig()
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        return [], _empty_selected_radar(radar)
    bootstrap_index = _first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        return [], _empty_selected_radar(radar)
    events = events[bootstrap_index:]
    initial = _initial_measurement(
        events[0],
        association="tracklet-viterbi",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
        radar_covariance_fn=radar_covariance_fn,
    )
    if initial is None:
        return [], _empty_selected_radar(radar)

    anchors = _build_rf_anchor_states(
        events=events,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )
    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        radar_covariance_fn=radar_covariance_fn,
    )
    records, accepted, replayed = _replay_selected_tracklet_path(
        events=events,
        selected_rows=selected,
        initial_measurement=initial,
        acceleration_std_mps2=acceleration_std_mps2,
        covariance=covariance,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
        radar_covariance_fn=radar_covariance_fn,
    )
    accepted_frame = _selected_rows_frame(radar, accepted)
    accepted_frame.attrs["attempted_selected_radar"] = _selected_rows_frame(radar, replayed)
    return records, accepted_frame


def _first_rf_bootstrap_index(events: list[dict[str, object]]) -> int | None:
    """Return the causal bootstrap index for non-oracle tracklet replay.

    The tracklet selector is truth-free and should not initialize from an
    arbitrary pre-RF radar candidate selected only by class probability.  If RF
    measurements are present, start at the first RF event and ignore earlier
    radar frames.  Radar-only inputs keep their historical radar bootstrap.
    """

    if not events:
        return None
    for index, event in enumerate(events):
        if event.get("kind") == "rf":
            return index
    return 0


def _build_rf_anchor_states(
    *,
    events: list[dict[str, object]],
    acceleration_std_mps2: float,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
) -> dict[int, _AnchorState]:
    """Return RF-only CV predictions at radar event indices."""

    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _robust_update_for_measurement,
    )

    tracker: AsyncConstantVelocityKalmanTracker | None = None
    anchors: dict[int, _AnchorState] = {}
    for event_index, event in enumerate(events):
        time_s = float(event["time_s"])
        if event["kind"] == "rf":
            measurement = event["measurement"]
            assert isinstance(measurement, TrackingMeasurement)
            if tracker is None:
                tracker = AsyncConstantVelocityKalmanTracker(
                    initial_position=measurement.vector,
                    initial_time_s=measurement.time_s,
                    acceleration_std_mps2=acceleration_std_mps2,
                )
            tracker.update(
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
        elif tracker is not None:
            tracker.predict_to(time_s)
            anchors[event_index] = _AnchorState(
                state=tracker.state.copy(),
                covariance=tracker.covariance_matrix.copy(),
            )
    return anchors


def _select_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, _AnchorState],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> list[pd.Series]:
    """Return selected radar rows from the lowest-cost Viterbi path.

    RaFT-UAV builds UAV/radar-specific candidate costs here, then delegates the
    generic dynamic-programming recursion to PyRecEst.
    """

    frames = [
        _nodes_for_radar_frame(
            event_index=i,
            candidates=event["candidates"],
            anchor=anchors.get(i),
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
            radar_covariance_fn=radar_covariance_fn,
        )
        for i, event in enumerate(events)
        if event["kind"] == "radar"
    ]
    if not frames:
        return []

    terminal_count = min(_soft_top_k_paths(config), len(frames[-1]))
    paths = _top_k_viterbi_paths_with_pyrecest(frames, config, terminal_count)
    if terminal_count <= 1:
        path_cost, path = paths[0]
        return _selected_rows_from_viterbi_path(path, path_cost, config)
    return _selected_rows_from_soft_viterbi_paths(paths, config)


def _top_k_viterbi_paths_with_pyrecest(
    frames: list[list[_ViterbiNode]],
    config: TrackletViterbiAssociationConfig,
    terminal_count: int,
) -> list[tuple[float, list[_ViterbiNode]]]:
    """Return RaFT-UAV path payloads from PyRecEst's generic Viterbi solver."""

    sequence_frames = [
        _sequence_candidates_for_frame(frame_position, frame, config)
        for frame_position, frame in enumerate(frames)
    ]

    def transition_cost(
        previous: SequenceAssociationNode,
        current: SequenceAssociationNode,
        context: SequenceTransitionContext,
    ) -> float:
        return _transition_cost(
            _node_from_sequence_candidate(previous),
            _node_from_sequence_candidate(current),
            config,
            previous_miss_streak=context.previous_miss_streak,
        )

    paths = solve_top_k_viterbi_sequence_associations(
        sequence_frames,
        transition_cost,
        top_k_terminal_paths=terminal_count,
    )
    return [
        (
            float(path.total_cost),
            [_node_from_sequence_candidate(node) for node in path.nodes],
        )
        for path in paths
    ]


def _sequence_candidates_for_frame(
    frame_position: int,
    frame: list[_ViterbiNode],
    config: TrackletViterbiAssociationConfig,
) -> list[SequenceAssociationNode]:
    return [
        SequenceAssociationNode(
            frame_index=node.event_index,
            candidate_index=None if node.is_miss else candidate_index,
            unary_cost=float(
                node.unary_cost
                + (config.missed_detection_cost if frame_position == 0 and node.is_miss else 0.0)
            ),
            is_missed_detection=node.is_miss,
            payload=node,
        )
        for candidate_index, node in enumerate(frame)
    ]


def _node_from_sequence_candidate(candidate: SequenceAssociationNode) -> _ViterbiNode:
    payload = candidate.payload
    if not isinstance(payload, _ViterbiNode):
        raise TypeError("sequence-association payload is not a RaFT-UAV Viterbi node")
    return payload


def _soft_top_k_paths(config: object) -> int:
    env_value = os.environ.get(_TRACKLET_SOFT_TOP_K_PATHS_ENV)
    if env_value:
        return max(1, int(env_value))
    return max(1, int(getattr(config, "soft_top_k_paths", 1)))


def _selected_rows_from_viterbi_path(
    path: Iterable[_ViterbiNode],
    path_cost: float,
    config: TrackletViterbiAssociationConfig,
) -> list[pd.Series]:
    """Return non-miss path rows annotated with miss-streak reacquisition terms."""

    rows: list[pd.Series] = []
    preceding_miss_streak = 0
    for node in path:
        if node.is_miss or node.row is None:
            preceding_miss_streak += 1
            continue
        row = node.row.copy()
        reacquisition_active = _reacquisition_is_active(preceding_miss_streak, node, config)
        reacquisition_cost = _reacquisition_cost(
            preceding_miss_streak,
            node,
            config,
        )
        if "association_mode" not in row.index or pd.isna(row.get("association_mode")):
            row["association_mode"] = "tracklet-viterbi"
        if "association_action" not in row.index or pd.isna(row.get("association_action")):
            row["association_action"] = "viterbi_selected"
        row["association_nis"] = float(node.anchor_nis)
        row["association_score"] = float(node.unary_cost)
        row["association_anchor_nis"] = float(node.anchor_nis)
        row["association_catprob_cost"] = float(node.catprob_cost)
        row["association_base_unary_cost"] = float(
            node.unary_cost if node.base_unary_cost is None else node.base_unary_cost
        )
        row["association_learned_candidate_cost"] = float(node.learned_cost)
        row["association_learned_candidate_probability"] = (
            np.nan if node.learned_probability is None else float(node.learned_probability)
        )
        row["association_candidate_score_mode"] = (
            str(config.learned_candidate_score_mode)
            if node.learned_probability is not None
            else "hand_tuned"
        )
        row["association_range_cost"] = float(node.range_cost)
        row["association_viterbi_path_cost"] = path_cost
        row["association_preceding_miss_streak"] = int(preceding_miss_streak)
        row["association_reacquisition_active"] = bool(reacquisition_active)
        row["association_reacquisition_cost"] = float(reacquisition_cost)
        row["association_reacquisition_gate_nis"] = (
            float(_reacquisition_effective_gate_nis(preceding_miss_streak, config))
            if reacquisition_active
            else np.nan
        )
        rows.append(row)
        preceding_miss_streak = 0
    return rows


def _selected_rows_from_soft_viterbi_paths(
    paths: list[tuple[float, list[_ViterbiNode]]],
    config: TrackletViterbiAssociationConfig,
) -> list[pd.Series]:
    """Moment-match radar positions from the best few full-sequence paths."""

    per_path_rows: list[tuple[float, list[pd.Series]]] = [
        (path_cost, _selected_rows_from_viterbi_path(path, path_cost, config))
        for path_cost, path in paths
    ]
    grouped: dict[tuple[str, int | float], list[tuple[float, pd.Series]]] = {}
    for path_cost, rows in per_path_rows:
        for row in rows:
            grouped.setdefault(_selected_row_event_key(row), []).append((path_cost, row))

    selected: list[pd.Series] = []
    for key in sorted(grouped, key=lambda item: (str(item[0]), item[1])):
        alternatives = grouped[key]
        if len(alternatives) == 1:
            row = alternatives[0][1].copy()
            row["association_soft_path_count"] = 1
            row["association_soft_path_weight_max"] = 1.0
            row["association_soft_path_weight_entropy"] = 0.0
            selected.append(row)
            continue
        costs = np.array([cost for cost, _ in alternatives], dtype=float)
        weights = _soft_path_weights(costs, config)
        frame = pd.DataFrame([row for _, row in alternatives]).reset_index(drop=True)
        positions = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
        mean = weights @ positions
        covariance = np.zeros((3, 3), dtype=float)
        for index, (_, row) in enumerate(alternatives):
            row_covariance = _row_position_covariance_or_default(row)
            diff = positions[index] - mean
            covariance += float(weights[index]) * (row_covariance + np.outer(diff, diff))
        best_index = int(np.argmin(costs))
        fused = alternatives[best_index][1].copy()
        fused["east_m"] = float(mean[0])
        fused["north_m"] = float(mean[1])
        fused["up_m"] = float(mean[2])
        fused["association_mode"] = "tracklet-viterbi"
        fused["association_action"] = "soft_viterbi_path_mixture"
        fused["association_soft_path_count"] = int(len(alternatives))
        fused["association_soft_path_weight_max"] = float(np.max(weights))
        fused["association_soft_path_weight_entropy"] = _weight_entropy(weights)
        fused["association_soft_path_temperature"] = _soft_path_temperature(config)
        fused["association_viterbi_path_cost"] = float(np.min(costs))
        fused["association_soft_path_cost_mean"] = float(weights @ costs)
        fused["association_cov_ee"] = float(covariance[0, 0])
        fused["association_cov_nn"] = float(covariance[1, 1])
        fused["association_cov_uu"] = float(covariance[2, 2])
        fused["association_cov_en"] = float(covariance[0, 1])
        fused["association_cov_eu"] = float(covariance[0, 2])
        fused["association_cov_nu"] = float(covariance[1, 2])
        fused["association_covariance_mode"] = "soft-viterbi-path-mixture"
        fused["association_cov_trace_m2"] = float(np.trace(covariance))
        selected.append(fused)
    return selected


def _nodes_for_radar_frame(
    *,
    event_index: int,
    candidates: pd.DataFrame,
    anchor: _AnchorState | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> list[_ViterbiNode]:
    from raft_uav.baselines.radar_association import _catprob_candidate_pool

    time_s = float(candidates["time_s"].median()) if "time_s" in candidates else float("nan")
    event_key = _radar_event_key(candidates)
    scored: list[tuple[float, _ViterbiNode]] = []
    for _, row in _catprob_candidate_pool(candidates, candidate_catprob_threshold).iterrows():
        position = _row_position(row)
        if position is None:
            continue
        row_covariance = _radar_covariance_for_row(row, covariance, radar_covariance_fn)
        anchor_nis, catprob_cost, range_cost = _candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=row_covariance,
            config=config,
        )
        base_unary_cost = float(config.anchor_nis_weight) * anchor_nis + catprob_cost + range_cost
        learned_cost, learned_probability = _learned_candidate_unary_cost(
            row=row,
            anchor=anchor,
            anchor_nis=anchor_nis,
            config=config,
        )
        unary_cost = _combine_candidate_cost(
            base_unary_cost,
            learned_cost,
            learned_probability,
            config,
        )
        node = _ViterbiNode(
            event_index=event_index,
            event_key=event_key,
            time_s=float(row.get("time_s", time_s)),
            row=row.copy(),
            position=position,
            velocity=_row_velocity(row),
            track_id=_optional_track_id(row.get("track_id")),
            unary_cost=float(unary_cost),
            anchor_nis=float(anchor_nis),
            catprob_cost=float(catprob_cost),
            range_cost=float(range_cost),
            learned_cost=float(learned_cost),
            learned_probability=learned_probability,
            base_unary_cost=float(base_unary_cost),
            has_anchor=bool(config.use_rf_anchor and anchor is not None),
        )
        scored.append((float(unary_cost), node))
    scored.sort(key=lambda item: item[0])
    nodes = [node for _, node in scored[: int(config.max_candidates_per_frame)]]
    nodes.append(
        _ViterbiNode(event_index, event_key, time_s, None, None, None, None, 0.0, 0.0, 0.0, 0.0, True)
    )
    return nodes


def _radar_covariance_for_row(
    row: pd.Series,
    covariance: np.ndarray,
    radar_covariance_fn: RadarCovarianceFn | None,
) -> np.ndarray:
    """Return the covariance to use for scoring or replaying one radar row."""

    if radar_covariance_fn is None:
        return np.asarray(covariance, dtype=float)
    return np.asarray(radar_covariance_fn(row, covariance), dtype=float)


def _candidate_cost_terms(
    *,
    row: pd.Series,
    position: np.ndarray,
    anchor: _AnchorState | None,
    covariance: np.ndarray,
    config: TrackletViterbiAssociationConfig,
) -> tuple[float, float, float]:
    anchor_nis = 0.0
    if config.use_rf_anchor and anchor is not None:
        candidate_covariance = np.asarray(covariance, dtype=float)
        try:
            from raft_uav.baselines.radar_association import _row_covariance

            row_covariance = _row_covariance(row)
            if row_covariance is not None:
                candidate_covariance = row_covariance
        except Exception:
            candidate_covariance = np.asarray(covariance, dtype=float)
        anchor_nis = _quadratic_form(
            position - np.asarray(anchor.state[:3], dtype=float),
            np.asarray(anchor.covariance[:3, :3], dtype=float) + candidate_covariance,
        )
    catprob = _optional_float(row.get("cat_prob_uav"))
    catprob = 1.0 if catprob is None else float(np.clip(catprob, config.min_catprob, 1.0))
    catprob_cost = float(config.catprob_weight) * float(-math.log(catprob))
    range_cost = 0.0
    range_m = _optional_float(row.get("range_m"))
    if config.range_gate_m is not None and range_m is not None:
        excess_m = max(0.0, float(range_m) - float(config.range_gate_m))
        if excess_m > 0.0:
            scale = max(float(config.range_gate_slack_m), 1.0)
            range_cost = float(config.range_penalty) * (excess_m / scale) ** 2
    return float(anchor_nis), float(catprob_cost), float(range_cost)


def _learned_candidate_unary_cost(
    *,
    row: pd.Series,
    anchor: _AnchorState | None,
    anchor_nis: float,
    config: TrackletViterbiAssociationConfig,
) -> tuple[float, float | None]:
    """Return learned per-candidate NLL cost from an existing association model."""

    model = config.learned_candidate_model
    if model is None or anchor is None:
        return 0.0, None
    candidate = pd.DataFrame([row.copy()])
    candidate["association_nis"] = float(anchor_nis)
    features = radar_association_feature_frame(
        candidate,
        tracker_state=np.asarray(anchor.state, dtype=float).reshape(6),
        current_track_id=None,
    )
    probability = float(model.predict_proba_features(features)[0])
    if not np.isfinite(probability):
        return 0.0, None
    probability = float(
        np.clip(probability, float(config.min_learned_candidate_probability), 1.0)
    )
    return float(-math.log(probability)), probability


def _combine_candidate_cost(
    base_unary_cost: float,
    learned_cost: float,
    learned_probability: float | None,
    config: TrackletViterbiAssociationConfig,
) -> float:
    if config.learned_candidate_model is None or learned_probability is None:
        return float(base_unary_cost)
    mode = str(config.learned_candidate_score_mode).strip().lower()
    if mode == "replace":
        return float(learned_cost)
    if mode == "additive":
        return float(base_unary_cost) + float(learned_cost)
    raise ValueError("learned_candidate_score_mode must be 'additive' or 'replace'")


def _transition_cost(
    previous: _ViterbiNode,
    current: _ViterbiNode,
    config: TrackletViterbiAssociationConfig,
    *,
    previous_miss_streak: int | None = None,
) -> float:
    """Return dynamic-programming transition cost between two radar nodes."""

    miss_streak = 1 if previous.is_miss else 0
    if previous_miss_streak is not None:
        miss_streak = max(0, int(previous_miss_streak))

    if current.is_miss:
        return float(config.missed_detection_cost) + (
            float(config.consecutive_miss_cost) if previous.is_miss else 0.0
        )
    if previous.is_miss or previous.position is None or current.position is None:
        return _reacquisition_cost(miss_streak, current, config)
    dt_s = max(float(current.time_s) - float(previous.time_s), 1.0e-3)
    predicted = previous.position if previous.velocity is None else previous.position + previous.velocity * dt_s
    position_std = float(config.transition_position_std_m) + float(config.transition_speed_std_mps) * dt_s
    motion_nis = float(np.sum(((current.position - predicted) / position_std) ** 2))
    displacement_velocity = (current.position - previous.position) / dt_s
    speed_excess = max(0.0, float(np.linalg.norm(displacement_velocity)) - float(config.max_speed_mps))
    speed_cost = float(config.max_speed_penalty) * (speed_excess / float(config.transition_speed_std_mps)) ** 2
    velocity_nis = 0.0
    if current.velocity is not None:
        velocity_nis += float(np.sum(((current.velocity - displacement_velocity) / config.velocity_std_mps) ** 2))
    if previous.velocity is not None and current.velocity is not None:
        velocity_nis += 0.25 * float(np.sum((((current.velocity - previous.velocity) / dt_s) / 8.0) ** 2))
    return float(
        config.transition_nis_weight * motion_nis
        + config.velocity_nis_weight * velocity_nis
        + speed_cost
        + _track_continuity_cost(previous.track_id, current.track_id, config)
        + _reacquisition_cost(miss_streak, current, config)
    )


def _reacquisition_is_active(
    previous_miss_streak: int,
    current: _ViterbiNode,
    config: TrackletViterbiAssociationConfig,
) -> bool:
    """Return whether RF-anchor reacquisition scoring applies to ``current``."""

    if current.is_miss or not current.has_anchor:
        return False
    return int(previous_miss_streak) >= int(config.reacquisition_miss_streak_threshold)


def _reacquisition_effective_gate_nis(
    previous_miss_streak: int,
    config: TrackletViterbiAssociationConfig,
) -> float:
    """Return the miss-streak widened RF-anchor NIS gate."""

    threshold = int(config.reacquisition_miss_streak_threshold)
    extra_misses = max(0, int(previous_miss_streak) - threshold)
    return float(config.reacquisition_gate_nis) * (
        1.0 + float(config.reacquisition_gate_growth) * float(extra_misses)
    )


def _reacquisition_cost(
    previous_miss_streak: int,
    current: _ViterbiNode,
    config: TrackletViterbiAssociationConfig,
) -> float:
    """Return miss-streak adaptive reacquisition cost around the RF anchor.

    After a miss streak, the ordinary Viterbi objective has no previous radar
    position to transition from.  This term uses the RF-only anchor as a search
    tube: candidates inside a widened NIS gate receive a bounded reward, while
    candidates outside the tube receive a smooth quadratic penalty.
    """

    if not _reacquisition_is_active(previous_miss_streak, current, config):
        return 0.0
    gate_nis = max(_reacquisition_effective_gate_nis(previous_miss_streak, config), 1.0e-9)
    anchor_nis = max(0.0, float(current.anchor_nis))
    closeness = max(0.0, 1.0 - anchor_nis / gate_nis)
    outside = max(0.0, (anchor_nis - gate_nis) / gate_nis)
    reward = float(config.reacquisition_reward) * closeness
    outside_penalty = float(config.reacquisition_outside_gate_penalty) * outside**2
    return float(outside_penalty - reward)


def _track_continuity_cost(
    previous_track_id: int | None,
    current_track_id: int | None,
    config: TrackletViterbiAssociationConfig,
) -> float:
    if previous_track_id is None:
        return 0.0
    if current_track_id is None:
        return float(config.missing_track_id_cost)
    return 0.0 if int(previous_track_id) == int(current_track_id) else float(config.track_switch_cost)


def _replay_selected_tracklet_path(
    *,
    events: list[dict[str, object]],
    selected_rows: list[pd.Series],
    initial_measurement: TrackingMeasurement,
    acceleration_std_mps2: float,
    covariance: np.ndarray,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    gate_thresholds_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
    radar_covariance_fn: RadarCovarianceFn | None = None,
) -> tuple[list[dict[str, object]], list[pd.Series], list[pd.Series]]:
    from raft_uav.baselines.radar_association import (
        _gate_threshold_for_measurement,
        _inflation_alpha_for_measurement,
        _max_residual_norm_for_measurement,
        _radar_row_to_measurement,
        _record,
        _robust_update_for_measurement,
    )
    from raft_uav.baselines.radar_update_policy import (
        apply_radar_update_policy,
        policy_record_fields,
    )

    selected_by_key = {_selected_row_event_key(row): row for row in selected_rows}
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=initial_measurement.vector,
        initial_time_s=initial_measurement.time_s,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    records: list[dict[str, object]] = []
    accepted_rows: list[pd.Series] = []
    replayed_rows: list[pd.Series] = []
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
        selected = selected_by_key.get(_radar_event_key(candidates))
        if selected is None:
            continue
        measurement_covariance = _radar_covariance_for_row(
            selected,
            covariance,
            radar_covariance_fn,
        )
        measurement = _radar_row_to_measurement(selected, measurement_covariance)
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
        replayed = selected.copy()
        replayed["association_replay_accepted"] = bool(diagnostics.accepted)
        replayed["association_replay_nis"] = float(diagnostics.nis)
        for key, value in policy_record_fields(selected).items():
            replayed[key] = value
        replayed_rows.append(replayed)
        if diagnostics.accepted:
            accepted_rows.append(replayed)
        record = _record(
            measurement,
            tracker,
            diagnostics,
            track_id=_optional_track_id(selected.get("track_id")),
            association_nis=_optional_float(selected.get("association_nis")),
            association_score=_optional_float(selected.get("association_score")),
            association_mode=str(selected.get("association_mode", "tracklet-viterbi")),
        )
        record.update(policy_record_fields(selected))
        records.append(record)
    return records, accepted_rows, replayed_rows


def _radar_event_key(candidates: pd.DataFrame) -> tuple[str, int | float]:
    if "frame_index" in candidates.columns and not candidates.empty:
        frame_index = _optional_float(candidates["frame_index"].iloc[0])
        if frame_index is not None:
            return "frame_index", int(frame_index)
    if "time_s" not in candidates.columns or candidates.empty:
        return "time_s", float("nan")
    return "time_s", round(float(pd.to_numeric(candidates["time_s"], errors="coerce").median()), 9)


def _selected_row_event_key(row: pd.Series) -> tuple[str, int | float]:
    frame_index = _optional_float(row.get("frame_index"))
    if frame_index is not None:
        return "frame_index", int(frame_index)
    time_s = _optional_float(row.get("time_s"))
    return ("time_s", float("nan")) if time_s is None else ("time_s", round(float(time_s), 9))


def _do_no_harm_decision(
    selected: pd.Series,
    measurement: TrackingMeasurement,
    gate_threshold: float | None,
):
    if not _env_flag(_DO_NO_HARM_RADAR_POLICY_ENV):
        return None
    from raft_uav.evaluation.fifth_wave_diagnostics import do_no_harm_radar_decision

    confidence = _optional_float(selected.get("association_learned_candidate_probability"))
    entropy = _optional_float(selected.get("association_soft_path_weight_entropy"))
    rf_anchor_nis = _optional_float(selected.get("association_anchor_nis"))
    rf_anchor_gate_nis = _optional_float(selected.get("association_reacquisition_gate_nis"))
    preceding_miss_streak = int(_optional_float(selected.get("association_preceding_miss_streak")) or 0)
    recent_recovery_mode = bool(selected.get("association_reacquisition_active", False))
    return do_no_harm_radar_decision(
        association_nis=_optional_float(selected.get("association_nis")),
        gate_threshold=gate_threshold,
        association_confidence=confidence,
        candidate_entropy=entropy,
        rf_anchor_nis=rf_anchor_nis,
        rf_anchor_gate_nis=rf_anchor_gate_nis,
        miss_streak=preceding_miss_streak,
        recent_recovery_mode=recent_recovery_mode,
    )


def _annotate_do_no_harm(selected: pd.Series, decision) -> pd.Series:
    if decision is None:
        return selected
    row = selected.copy()
    row["do_no_harm_action"] = decision.action
    row["do_no_harm_risk_score"] = float(decision.risk_score)
    row["do_no_harm_reasons"] = ";".join(decision.reasons)
    row["do_no_harm_covariance_scale"] = float(decision.covariance_scale)
    row["do_no_harm_defer_lag_s"] = float(decision.defer_lag_s)
    return row


def _scaled_tracking_measurement(
    measurement: TrackingMeasurement,
    scale: float,
) -> TrackingMeasurement:
    covariance = np.asarray(measurement.covariance, dtype=float)
    return TrackingMeasurement(
        time_s=measurement.time_s,
        vector=measurement.vector,
        covariance=covariance * float(scale),
        source=measurement.source,
    )


def _soft_path_temperature(config: object) -> float:
    env_value = os.environ.get(_TRACKLET_SOFT_PATH_TEMPERATURE_ENV)
    if env_value:
        return max(float(env_value), 1.0e-9)
    return max(float(getattr(config, "soft_path_temperature", 1.0)), 1.0e-9)


def _soft_path_weights(costs: np.ndarray, config: object) -> np.ndarray:
    temperature = _soft_path_temperature(config)
    values = -np.asarray(costs, dtype=float).reshape(-1) / temperature
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return np.full(values.size, 1.0 / max(values.size, 1))
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.size, 1.0 / max(values.size, 1))
    return weights / total


def _row_position_covariance_or_default(row: pd.Series) -> np.ndarray:
    try:
        from raft_uav.baselines.radar_association import _row_covariance

        covariance = _row_covariance(row)
        if covariance is not None:
            return covariance
    except Exception:
        pass
    return np.diag([25.0**2, 25.0**2, 35.0**2])


def _weight_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-300, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _row_position(row: pd.Series) -> np.ndarray | None:
    try:
        position = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
    except (KeyError, TypeError, ValueError):
        return None
    return position if np.isfinite(position).all() else None


def _row_velocity(row: pd.Series) -> np.ndarray | None:
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


def _quadratic_form(residual: np.ndarray, covariance: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    covariance = np.asarray(covariance, dtype=float)
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ residual
    value = float(residual.T @ solved)
    return value if np.isfinite(value) else 0.0
