from __future__ import annotations

import pandas as pd

from raft_uav.research.tracklet_models import tracklet_feature_frame


def test_tracklet_feature_frame_sorts_serialized_frame_keys_numerically() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [1, 1, 1],
            "frame_index": ["1", "10", "2"],
            "time_s": ["1", "10", "2"],
            "east_m": [1.0, 10.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    features = tracklet_feature_frame(radar, max_frame_gap=5.0)

    assert features["frames"].tolist() == [2, 1]
    assert features["start_time_s"].tolist() == [1.0, 10.0]
    assert features["end_time_s"].tolist() == [2.0, 10.0]
    assert features["start_east_m"].tolist() == [1.0, 10.0]
    assert features["end_east_m"].tolist() == [2.0, 10.0]
