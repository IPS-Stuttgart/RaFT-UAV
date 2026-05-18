import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_fixed_lag import (
    run_async_cv_baseline_with_fixed_lag_tracklet_viterbi_association_and_replay,
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
