from __future__ import annotations

import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_track_features_preserve_duplicate_index_rows_and_order() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0, 0.0, 0.0],
            "frame_index": [1, 0, 0],
            "track_id": [7, 8, 7],
            "east_m": [1.0, 10.0, 0.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.8, 0.4, 0.2],
        },
        index=pd.Index([5, 5, 9], name="radar_row"),
    )

    featured = add_track_level_features(radar, window_frames=2)

    assert len(featured) == len(radar)
    assert featured.index.equals(radar.index)
    assert featured["track_id"].tolist() == [7, 8, 7]
    assert featured["track_age_frames"].tolist() == [1.0, 0.0, 0.0]
    assert featured["track_catprob_mean_window"].tolist() == [0.5, 0.4, 0.2]
