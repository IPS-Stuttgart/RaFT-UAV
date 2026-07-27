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


def _partially_indexed_radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [10.0, 10.0, np.nan],
            "track_id": [1, 2, 1],
            "time_s": [1.0, 1.0, 2.0],
            "east_m": [1.0, 100.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.1, 0.9],
        }
    )


def test_radar_association_preserves_frames_with_missing_frame_index() -> None:
    radar = _partially_indexed_radar()

    groups = _radar_frame_groups(radar)
    assert [group["time_s"].iloc[0] for group in groups] == [1.0, 2.0]
    assert [len(group) for group in groups] == [2, 1]

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[_rf_measurement(0.0, 0.0)],
        radar=radar,
        association="prediction-nis",
        candidate_catprob_threshold=None,
    )

    assert [record["source"] for record in records] == ["rf", "radar", "radar"]
    assert selected["time_s"].tolist() == [1.0, 2.0]
    assert selected["track_id"].tolist() == [1, 1]
