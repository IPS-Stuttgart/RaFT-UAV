import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def _featured_frames(frame_indices: list[int]) -> pd.DataFrame:
    radar = pd.DataFrame(
        {
            "track_id": [7] * len(frame_indices),
            "time_s": np.arange(len(frame_indices), dtype=float),
            "frame_index": frame_indices,
        }
    )
    return add_track_level_features(radar)


def _feature_streak(frame_indices: list[int]) -> np.ndarray:
    return _featured_frames(frame_indices)["track_hit_streak_frames"].to_numpy(dtype=float)


def test_hit_streak_resets_when_frame_counter_moves_backwards() -> None:
    np.testing.assert_array_equal(
        _feature_streak([10, 11, 0, 1]),
        np.array([1.0, 2.0, 1.0, 2.0]),
    )


def test_hit_streak_does_not_count_duplicate_frame_twice() -> None:
    np.testing.assert_array_equal(
        _feature_streak([10, 11, 11, 12]),
        np.array([1.0, 2.0, 2.0, 3.0]),
    )


def test_track_age_does_not_count_duplicate_frame_twice() -> None:
    age = _featured_frames([10, 11, 11, 12])["track_age_frames"].to_numpy(dtype=float)

    np.testing.assert_array_equal(age, np.array([0.0, 1.0, 1.0, 2.0]))
