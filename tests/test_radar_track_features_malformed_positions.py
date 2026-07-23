from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_track_features_coerce_malformed_position_cells_to_missing() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "frame_index": [0, 1, 2],
            "track_id": [7, 7, 7],
            "east_m": ["0.0", "malformed", "2.0"],
            "north_m": ["0.0", "0.0", "0.0"],
            "up_m": ["0.0", "0.0", "0.0"],
        }
    )

    featured = add_track_level_features(radar)

    np.testing.assert_allclose(
        featured["track_position_step_m"].to_numpy(dtype=float),
        np.array([0.0, np.nan, np.nan]),
        equal_nan=True,
    )
    assert featured["track_speed_from_positions_mps"].isna().all()
    assert featured["track_range_rate_mps"].isna().all()
