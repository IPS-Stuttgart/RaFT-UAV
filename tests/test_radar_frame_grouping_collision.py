from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import (
    _radar_frame_groups,
    run_async_cv_baseline_with_radar_association,
)


def _rf_measurement(time_s: float, east_m: float) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([east_m, 0.0]),
        covariance=np.diag([1.0, 1.0]),
        source="rf",
    )


def test_partial_frame_indices_preserve_same_timestamp_frame_collisions() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0.0, 1.0, np.nan],
            "track_id": [1, 2, 3],
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [0.0, 100.0, 1.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8, 0.7],
        }
    )

    groups = _radar_frame_groups(radar)
    assert [group["track_id"].tolist() for group in groups] == [[1], [2], [3]]

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(-1.0, 0.0)],
        radar=radar,
        association="prediction-nis",
        candidate_catprob_threshold=None,
    )

    assert [record["source"] for record in records] == ["rf", "radar", "radar", "radar"]
    assert selected["track_id"].tolist() == [1, 2, 3]
