"""Fixed-lag tracklet-Viterbi radar association.

The offline tracklet-Viterbi baseline selects a single path over the full flight.
This module keeps the same scoring objective but constrains each committed radar
decision to use only a bounded look-ahead window. It is still an offline batch
implementation for experiments, but each individual committed decision is made
with at most ``lag_s`` seconds of future radar/RF information.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _build_rf_anchor_states,
    _radar_event_key,
    _select_tracklet_viterbi_path,
    _selected_row_event_key,
)
from raft_uav.baselines.tracklet_viterbi_result import (
    _empty_replayed_rows,
    _replay_selected_tracklet_path_with_replay,
)


def run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    lag_s: float,
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
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    """Run fixed-lag tracklet-Viterbi and replay committed radar choices."""

    if lag_s <= 0.0:
        raise ValueError("lag_s must be positive")

    from raft_uav.baselines.radar_association import (
        _empty_selected_radar,
        _events,
        _initial_measurement,
        _selected_rows_frame,
    )

    cfg = config or TrackletViterbiAssociationConfig()
    covariance = np.diag(
        [
            float(radar_xy_std_m) ** 2,
            float(radar_xy_std_m) ** 2,
            float(radar_z_std_m) ** 2,
        ]
    )
    events = _events(list(rf_measurements), radar)
    if not events:
        empty = _empty_selected_radar(radar)
        return [], empty, _empty_replayed_rows(empty)

    initial = _initial_measurement(
        events[0],
        association="tracklet-viterbi-fixed-lag",
        covariance=covariance,
        truth=None,
        truth_gate_m=150.0,
        truth_time_gate_s=1.0,
    )
    if initial is None:
        empty = _empty_selected_radar(radar)
        return [], empty, _empty_replayed_rows(empty)

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
    selected = select_fixed_lag_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        lag_s=lag_s,
    )
    records, accepted, replayed = _replay_selected_tracklet_path_with_replay(
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
    )
    accepted_frame = _selected_rows_frame(radar, accepted)
    replayed_frame = _selected_rows_frame(radar, replayed)
    return records, accepted_frame, replayed_frame


def select_fixed_lag_tracklet_viterbi_path(
    *,
    events: list[dict[str, object]],
    anchors: Mapping[int, object],
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    lag_s: float,
) -> list[pd.Series]:
    """Commit each radar frame using only a bounded future Viterbi window.

    For each radar event at time ``t``, the function solves the ordinary
    Viterbi objective on events with ``t <= time <= t + lag_s`` and commits only
    the decision for the first radar event in that window. This implements a
    fixed-lag look-ahead diagnostic without using full-flight future context.
    """

    if lag_s <= 0.0:
        raise ValueError("lag_s must be positive")

    radar_indices = [
        index for index, event in enumerate(events) if event.get("kind") == "radar"
    ]
    committed: dict[tuple[str, int | float], pd.Series] = {}
    for global_index in radar_indices:
        candidates = events[global_index]["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        event_key = _radar_event_key(candidates)
        if event_key in committed:
            continue

        start_s = float(events[global_index]["time_s"])
        end_s = start_s + float(lag_s)
        window_indices = [
            index
            for index, event in enumerate(events)
            if start_s <= float(event["time_s"]) <= end_s
        ]
        if global_index not in window_indices:
            window_indices.append(global_index)
            window_indices.sort()

        local_events = [events[index] for index in window_indices]
        local_anchors = {
            local_index: anchors[global_index]
            for local_index, global_index in enumerate(window_indices)
            if global_index in anchors
        }
        selected_window = _select_tracklet_viterbi_path(
            events=local_events,
            anchors=local_anchors,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
        )
        selected_by_key = {
            _selected_row_event_key(row): row for row in selected_window
        }
        selected = selected_by_key.get(event_key)
        if selected is None:
            continue

        row = selected.copy()
        row["association_mode"] = "tracklet-viterbi-fixed-lag"
        row["association_lag_s"] = float(lag_s)
        row["association_lag_window_start_s"] = start_s
        row["association_lag_window_end_s"] = end_s
        row["association_lag_window_event_count"] = int(len(local_events))
        row["association_lag_window_radar_count"] = int(
            sum(event.get("kind") == "radar" for event in local_events)
        )
        committed[event_key] = row

    return list(committed.values())
