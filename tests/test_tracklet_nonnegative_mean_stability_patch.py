"""Regression coverage for finite tracklet means whose naive sums overflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.tracklet_models import (
    frame_context_features,
    tracklet_feature_frame,
)


def test_tracklet_feature_means_stay_finite_when_naive_sum_overflows() -> None:
    magnitude = 1.0e308
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0, 1, 2],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [-magnitude, 0.0, magnitude],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        features = tracklet_feature_frame(radar)

    np.testing.assert_allclose(features["mean_speed_mps"], [magnitude])
    np.testing.assert_allclose(features["max_speed_mps"], [magnitude])
    np.testing.assert_allclose(
        features["mean_range_m"],
        [magnitude * (2.0 / 3.0)],
    )
    np.testing.assert_allclose(features["range_span_m"], [magnitude])


def test_frame_context_neighbor_means_stay_finite_when_sum_overflows() -> None:
    magnitude = 1.0e308
    candidates = pd.DataFrame(
        {
            "east_m": [0.0, magnitude, 0.0],
            "north_m": [0.0, 0.0, magnitude],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        featured = frame_context_features(candidates)

    diagonal = np.hypot(magnitude, magnitude)
    outer_mean = 0.5 * magnitude + 0.5 * diagonal
    np.testing.assert_allclose(
        featured["mean_neighbor_distance_m"].to_numpy(dtype=float),
        [magnitude, outer_mean, outer_mean],
    )
