from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.tracklet_models import (
    frame_context_features,
    tracklet_feature_frame,
)


def test_tracklet_features_keep_representable_large_norms_finite() -> None:
    magnitude = 1.0e308
    radar = pd.DataFrame(
        {
            "track_id": [7, 7],
            "frame_index": [0, 1],
            "time_s": [0.0, 2.0],
            "east_m": [0.0, magnitude],
            "north_m": [0.0, magnitude],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        features = tracklet_feature_frame(radar)

    expected_norm = np.hypot(magnitude, magnitude)
    expected_speed = expected_norm / 2.0
    assert np.isfinite(expected_norm)
    np.testing.assert_allclose(features["mean_speed_mps"], [expected_speed])
    np.testing.assert_allclose(features["max_speed_mps"], [expected_speed])
    np.testing.assert_allclose(features["mean_range_m"], [expected_norm / 2.0])
    np.testing.assert_allclose(features["range_span_m"], [expected_norm])


def test_frame_context_keeps_representable_large_neighbor_distances_finite() -> None:
    magnitude = 1.0e308
    candidates = pd.DataFrame(
        {
            "east_m": [0.0, magnitude],
            "north_m": [0.0, magnitude],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        featured = frame_context_features(candidates)

    expected_norm = np.hypot(magnitude, magnitude)
    np.testing.assert_allclose(
        featured["nearest_neighbor_distance_m"].to_numpy(dtype=float),
        [expected_norm, expected_norm],
    )
    np.testing.assert_allclose(
        featured["mean_neighbor_distance_m"].to_numpy(dtype=float),
        [expected_norm, expected_norm],
    )
