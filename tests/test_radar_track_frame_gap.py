import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_frame_gap_restarts_after_frame_counter_reset() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7, 7],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "frame_index": [10, 11, 0, 2],
        }
    )

    featured = add_track_level_features(radar)

    np.testing.assert_array_equal(
        featured["track_frame_gap"].to_numpy(dtype=float),
        np.array([0.0, 1.0, 0.0, 2.0]),
    )
