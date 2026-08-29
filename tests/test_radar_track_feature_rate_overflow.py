from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_position_speed_survives_unrepresentable_raw_step() -> None:
    magnitude = 1.0e308
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 4.0],
            "frame_index": [0, 1],
            "track_id": [7, 7],
            "east_m": [-magnitude, magnitude],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        featured = add_track_level_features(radar, window_frames=2)

    assert np.isnan(float(featured.loc[1, "track_position_step_m"]))
    assert float(featured.loc[1, "track_speed_from_positions_mps"]) == 5.0e307
