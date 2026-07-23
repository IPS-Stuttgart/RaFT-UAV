from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.radar_track_features import add_track_level_features


def _radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "time_s": [0.0, 1.0, 2.0],
            "frame_index": [0, 1, 2],
            "cat_prob_uav": [0.0, 1.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    "window_frames",
    [
        0,
        1.5,
        True,
        np.bool_(False),
        np.nan,
        np.array([2]),
        np.ma.masked,
    ],
)
def test_track_features_reject_malformed_window_sizes(window_frames: object) -> None:
    with pytest.raises(ValueError, match="window_frames must be a positive integer"):
        add_track_level_features(_radar(), window_frames=window_frames)


@pytest.mark.parametrize("window_frames", ["2", 2.0, np.array(2)])
def test_track_features_normalize_integer_like_window_sizes(
    window_frames: object,
) -> None:
    featured = add_track_level_features(_radar(), window_frames=window_frames)

    np.testing.assert_allclose(
        featured["track_catprob_mean_window"],
        np.array([0.0, 0.5, 1.0]),
    )
