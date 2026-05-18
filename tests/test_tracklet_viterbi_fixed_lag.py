import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import _events
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_fixed_lag import (
    run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay,
    select_fixed_lag_tracklet_viterbi_path,
)


def _rf_measurement(time_s: float, east_m: float, north_m: float = 0.0) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, north_m]),
        covariance=np.diag([4.0, 4.0]),
        source="rf",
    )


def test_fixed_lag_tracklet_viterbi_uses_bounded_future_window():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.8,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 1000.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )

    _, _, viterbi_selected = (
        run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay(
            rf_measurements=[_rf_measurement(0.0, 0.0), _rf_measurement(1.0, 10.0)],
            radar=radar,
            lag_s=1.0,
            candidate_catprob_threshold=None,
            config=TrackletViterbiAssociationConfig(
                anchor_nis_weight=2.0,
                track_switch_cost=20.0,
                max_speed_penalty=10_000.0,
            ),
        )
    )

    assert viterbi_selected["track_id"].tolist() == [1, 1]
    assert viterbi_selected["association_mode"].unique().tolist() == [
        "tracklet-viterbi-fixed-lag"
    ]
    assert viterbi_selected["association_lag_s"].tolist() == [1.0, 1.0]


def test_fixed_lag_tracklet_viterbi_conditions_on_previous_committed_choice():
    radar = pd.DataFrame(
        [
            {
                "frame_index": 0,
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 0,
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.3,
            },
            {
                "frame_index": 1,
                "track_id": 2,
                "time_s": 1.0,
                "east_m": 110.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
            },
        ]
    )
    events = _events([], radar)
    covariance = np.diag([25.0**2, 25.0**2, 35.0**2])

    selected = select_fixed_lag_tracklet_viterbi_path(
        events=events,
        anchors={},
        covariance=covariance,
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(
            catprob_weight=2.0,
            track_switch_cost=1_000.0,
            max_speed_mps=200.0,
            max_speed_penalty=0.0,
            transition_nis_weight=0.0,
            velocity_nis_weight=0.0,
            anchor_nis_weight=0.0,
        ),
        lag_s=1.0,
    )

    assert [int(row["track_id"]) for row in selected] == [1, 1]
    assert bool(selected[1]["association_prefix_adjusted"])
    assert float(selected[1]["association_prefix_adjusted_cost"]) < float(
        selected[1]["association_prefix_unconstrained_cost"]
    )
