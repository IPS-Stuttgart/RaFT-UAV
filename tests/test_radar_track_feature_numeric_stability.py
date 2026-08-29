from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def test_track_features_keep_representable_large_norms_finite() -> None:
    magnitude = 1.0e308
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 2.0],
            "frame_index": [0, 1],
            "track_id": [7, 7],
            "east_m": [0.0, magnitude],
            "north_m": [0.0, magnitude],
            "up_m": [0.0, 0.0],
            "velocity_east_mps": [0.0, magnitude],
            "velocity_north_mps": [0.0, magnitude],
            "velocity_down_mps": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        featured = add_track_level_features(radar, window_frames=2)

    expected_norm = np.hypot(magnitude, magnitude)
    expected_rate = expected_norm / 2.0
    assert np.isfinite(expected_norm)
    np.testing.assert_allclose(
        featured["track_position_step_m"].to_numpy(dtype=float),
        np.array([0.0, expected_norm]),
    )
    np.testing.assert_allclose(
        featured["track_speed_from_positions_mps"].to_numpy(dtype=float),
        np.array([np.nan, expected_rate]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        featured["track_range_rate_mps"].to_numpy(dtype=float),
        np.array([np.nan, expected_rate]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        featured["track_velocity_smoothness_mps"].to_numpy(dtype=float),
        np.array([np.nan, expected_norm]),
        equal_nan=True,
    )


def test_track_features_ignore_nonfinite_time_intervals() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, np.inf],
            "frame_index": [0, 1],
            "track_id": [7, 7],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        featured = add_track_level_features(radar, window_frames=2)

    np.testing.assert_allclose(
        featured["track_position_step_m"].to_numpy(dtype=float),
        np.array([0.0, 10.0]),
    )
    np.testing.assert_allclose(
        featured["track_speed_from_positions_mps"].to_numpy(dtype=float),
        np.array([np.nan, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        featured["track_range_rate_mps"].to_numpy(dtype=float),
        np.array([np.nan, np.nan]),
        equal_nan=True,
    )


def test_track_features_ignore_finite_difference_overflow() -> None:
    magnitude = 1.0e308
    radar = pd.DataFrame(
        {
            "time_s": [-magnitude, magnitude],
            "frame_index": [-magnitude, magnitude],
            "track_id": [7, 7],
            "range_m": [1.0, 2.0],
            "velocity_east_mps": [-magnitude, magnitude],
            "velocity_north_mps": [0.0, 0.0],
            "velocity_down_mps": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        featured = add_track_level_features(radar, window_frames=2)

    np.testing.assert_allclose(
        featured["track_age_frames"].to_numpy(dtype=float),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        featured["track_hit_streak_frames"].to_numpy(dtype=float),
        np.array([1.0, 1.0]),
    )
    np.testing.assert_allclose(
        featured["track_time_since_first_s"].to_numpy(dtype=float),
        np.array([0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        featured["track_frame_gap"].to_numpy(dtype=float),
        np.array([0.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        featured["track_range_rate_mps"].to_numpy(dtype=float),
        np.array([np.nan, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        featured["track_velocity_smoothness_mps"].to_numpy(dtype=float),
        np.array([np.nan, np.nan]),
        equal_nan=True,
    )
