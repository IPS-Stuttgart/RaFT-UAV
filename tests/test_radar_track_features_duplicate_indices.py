from __future__ import annotations

import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_track_features_preserve_duplicate_input_indices_and_row_order() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 0.0],
            "frame_index": [0, 1, 0],
            "track_index": [0, 0, 1],
            "track_id": [7, 7, 8],
            "east_m": [0.0, 1.0, 10.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        },
        index=[5, 5, 9],
    )

    output = add_track_level_features(radar)

    assert len(output) == len(radar)
    assert output.index.tolist() == [5, 5, 9]
    assert output["time_s"].tolist() == [0.0, 1.0, 0.0]
    assert output["track_age_frames"].tolist() == [0.0, 1.0, 0.0]
