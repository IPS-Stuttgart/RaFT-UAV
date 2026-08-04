from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


_DERIVED_COLUMNS = [
    "track_age_frames",
    "track_hit_streak_frames",
    "track_time_since_first_s",
    "track_frame_gap",
    "track_position_step_m",
    "track_speed_from_positions_mps",
    "track_range_rate_mps",
]


def _radar_rows(timestamps: list[object]) -> pd.DataFrame:
    positions = np.asarray([1.0, 10.0, 2.0])
    return pd.DataFrame(
        {
            "time_s": timestamps,
            "track_id": [7, 7, 7],
            "east_m": positions,
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "range_m": positions,
        }
    )


def test_track_features_sort_numeric_string_timestamps_chronologically() -> None:
    numeric = add_track_level_features(_radar_rows([1.0, 10.0, 2.0]))
    text = add_track_level_features(_radar_rows(["1", "10", "2"]))

    np.testing.assert_allclose(
        text[_DERIVED_COLUMNS].to_numpy(dtype=float),
        numeric[_DERIVED_COLUMNS].to_numpy(dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        text["track_speed_from_positions_mps"].to_numpy(dtype=float),
        np.asarray([np.nan, 1.0, 1.0]),
        equal_nan=True,
    )
