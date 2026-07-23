from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_missing_track_ids_do_not_share_temporal_features() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "frame_index": [10, 11],
            "track_id": [np.nan, np.nan],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.2, 0.8],
        }
    )

    featured = add_track_level_features(radar, window_frames=10)

    np.testing.assert_array_equal(featured["track_age_frames"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(featured["track_hit_streak_frames"], np.array([1.0, 1.0]))
    np.testing.assert_array_equal(featured["track_frame_gap"], np.array([0.0, 0.0]))
    np.testing.assert_array_equal(featured["track_position_step_m"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(featured["track_catprob_mean_window"], np.array([0.2, 0.8]))
    assert featured["track_speed_from_positions_mps"].isna().all()
