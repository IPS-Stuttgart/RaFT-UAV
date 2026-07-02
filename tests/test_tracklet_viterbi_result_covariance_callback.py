from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_range_covariance import _range_adaptive_covariance_fn
from raft_uav.baselines.tracklet_viterbi_result import (
    run_async_cv_baseline_with_tracklet_viterbi_result,
)


def test_result_api_propagates_range_adaptive_covariance_callback() -> None:
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    radar_covariance_fn = _range_adaptive_covariance_fn(config)
    rf_measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([0.0, 0.0, 0.0]),
            covariance=np.eye(3),
            source="rf",
        )
    ]
    radar = pd.DataFrame(
        [
            {
                "time_s": 1.0,
                "frame_index": 1,
                "track_id": 7,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.99,
                "range_m": 1200.0,
            }
        ]
    )

    result = run_async_cv_baseline_with_tracklet_viterbi_result(
        rf_measurements=rf_measurements,
        radar=radar,
        candidate_catprob_threshold=None,
        config=config,
        radar_covariance_fn=radar_covariance_fn,
    )

    assert not result.viterbi_selected_radar.empty
    selected = result.viterbi_selected_radar.iloc[0]
    ledger = result.radar_candidate_ledger.iloc[0]
    assert np.isclose(selected["association_radar_xy_std_m"], 42.0)
    assert np.isclose(selected["association_radar_z_std_m"], 60.0)
    assert bool(selected["association_radar_covariance_adaptive"])
    assert np.isclose(ledger["association_radar_xy_std_m"], 42.0)
    assert np.isclose(ledger["association_radar_z_std_m"], 60.0)
