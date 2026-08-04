import numpy as np
import pandas as pd

import raft_uav.baselines.imm_radar_association as imm_radar_association
from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker
from raft_uav.baselines.imm_radar_association import (
    _imm_scored_candidates,
    run_async_imm_baseline_with_radar_association,
)
from raft_uav.baselines.kalman import TrackingMeasurement


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([1.0, 1.0]),
        source="rf",
    )


def test_imm_scored_candidates_ignores_invalid_candidate_positions():
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )
    candidates = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": np.nan,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 2.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.1,
            },
        ]
    )

    scored = _imm_scored_candidates(candidates, tracker=tracker, base_covariance=np.eye(3))

    assert scored["track_id"].tolist() == [1]
    assert scored["association_candidate_rows"].tolist() == [2]
    assert scored["association_invalid_candidate_rows"].tolist() == [1]
    assert np.isfinite(scored["association_score"].iloc[0])


def test_imm_scored_candidates_returns_empty_for_all_invalid_positions():
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )
    candidates = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": np.nan,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            }
        ]
    )

    scored = _imm_scored_candidates(candidates, tracker=tracker, base_covariance=np.eye(3))

    assert scored.empty


def test_imm_candidate_selection_handles_duplicate_index_labels():
    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )
    candidates = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 2.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
        ],
        index=[7, 7],
    )

    selected = imm_radar_association._select_imm_radar_candidate(
        candidates,
        tracker=tracker,
        base_covariance=np.eye(3),
        association="imm-mixture-nis",
        candidate_catprob_threshold=None,
        rf_measurements=[],
        rf_anchor_weight=0.35,
        rf_anchor_time_gate_s=2.0,
        rf_anchor_nis_cap=25.0,
    )

    assert isinstance(selected, pd.Series)
    assert selected["track_id"] == 2


def test_imm_radar_association_ignores_invalid_candidate_positions():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 99,
                "time_s": 2.0,
                "east_m": np.nan,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 2.0,
                "east_m": 2.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.1,
            },
        ]
    )

    _records, selected = run_async_imm_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 1.0)],
        radar=radar,
        candidate_catprob_threshold=None,
    )

    assert selected["track_id"].tolist() == [1]
    assert selected["association_candidate_rows"].tolist() == [2]
    assert selected["association_invalid_candidate_rows"].tolist() == [1]
