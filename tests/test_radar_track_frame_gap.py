import numpy as np
import pandas as pd

from raft_uav.baselines.radar_track_features import add_track_level_features


def _feature_gap(frame_indices: list[int]) -> np.ndarray:
    radar = pd.DataFrame(
        {
            "track_id": [7] * len(frame_indices),
            "time_s": np.arange(len(frame_indices), dtype=float),
            "frame_index": frame_indices,
        }
    )
    featured = add_track_level_features(radar)
    return featured["track_frame_gap"].to_numpy(dtype=float)


def test_frame_gap_resets_when_frame_counter_moves_backwards() -> None:
    np.testing.assert_array_equal(
        _feature_gap([10, 11, 0, 1]),
        np.array([0.0, 1.0, 0.0, 1.0]),
    )


def test_frame_gap_preserves_duplicate_and_forward_deltas() -> None:
    np.testing.assert_array_equal(
        _feature_gap([10, 10, 12]),
        np.array([0.0, 0.0, 2.0]),
    )
