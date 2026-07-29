from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_velocity_smoothness_keeps_unobserved_deltas_missing() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "frame_index": [0, 1, 2, 3],
            "track_id": [7, 7, 7, 7],
            "velocity_east_mps": [0.0, 1.0, "malformed", 2.0],
            "velocity_north_mps": [0.0, 0.0, 0.0, 0.0],
            "velocity_down_mps": [0.0, 0.0, 0.0, 0.0],
        }
    )

    featured = add_track_level_features(radar, window_frames=3)

    np.testing.assert_allclose(
        featured["track_velocity_smoothness_mps"].to_numpy(dtype=float),
        np.array([np.nan, 1.0, np.nan, np.nan]),
        equal_nan=True,
    )
