from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_track_features_reset_when_track_ids_are_reused_across_sequences() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "track_id": [7, 7, 7, 7],
            "time_s": [0.0, 1.0, 10.0, 11.0],
            "frame_index": [0, 1, 0, 1],
            "east_m": [0.0, 1.0, 100.0, 101.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "cat_prob_uav": [0.2, 0.4, 0.8, 1.0],
        }
    )

    featured = add_track_level_features(radar, window_frames=10)
    second_sequence = featured.loc[featured["sequence_id"] == "flight-b"]

    np.testing.assert_array_equal(
        second_sequence["track_age_frames"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        second_sequence["track_hit_streak_frames"].to_numpy(dtype=float),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        second_sequence["track_time_since_first_s"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        second_sequence["track_position_step_m"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    assert np.isnan(second_sequence["track_speed_from_positions_mps"].iloc[0])
    assert second_sequence["track_speed_from_positions_mps"].iloc[1] == 1.0
    np.testing.assert_allclose(
        second_sequence["track_catprob_mean_window"].to_numpy(dtype=float),
        np.array([0.8, 0.9]),
    )


def test_track_features_reset_when_track_ids_are_reused_across_flight_ids() -> None:
    radar = pd.DataFrame(
        {
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "track_id": [7, 7, 7, 7],
            "time_s": [0.0, 1.0, 10.0, 11.0],
            "frame_index": [0, 1, 0, 1],
            "east_m": [0.0, 1.0, 100.0, 101.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "cat_prob_uav": [0.2, 0.4, 0.8, 1.0],
        }
    )

    featured = add_track_level_features(radar, window_frames=10)
    second_flight = featured.loc[featured["flight_id"] == "flight-b"]

    np.testing.assert_array_equal(
        second_flight["track_age_frames"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        second_flight["track_hit_streak_frames"].to_numpy(dtype=float),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_array_equal(
        second_flight["track_time_since_first_s"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_array_equal(
        second_flight["track_position_step_m"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    assert np.isnan(second_flight["track_speed_from_positions_mps"].iloc[0])
    assert second_flight["track_speed_from_positions_mps"].iloc[1] == 1.0
    np.testing.assert_allclose(
        second_flight["track_catprob_mean_window"].to_numpy(dtype=float),
        np.array([0.8, 0.9]),
    )
