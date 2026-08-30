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


def test_tracklet_features_keep_large_finite_means_finite() -> None:
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
    np.testing.assert_allclose(features["mean_range_m"], [magnitude * (2.0 / 3.0)])
    np.testing.assert_allclose(features["range_span_m"], [magnitude])


def test_tracklet_features_ignore_zero_duration_steps_in_speed_summary() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0, 1, 2],
            "time_s": [0.0, 0.0, 2.0],
            "east_m": [0.0, 100.0, 104.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        features = tracklet_feature_frame(radar)

    np.testing.assert_allclose(features["mean_speed_mps"], [2.0])
    np.testing.assert_allclose(features["max_speed_mps"], [2.0])


def test_tracklet_features_recover_representable_speed_after_step_overflow() -> None:
    magnitude = 1.0e308
    radar = pd.DataFrame(
        {
            "track_id": [7, 7],
            "frame_index": [0, 1],
            "time_s": [0.0, 4.0],
            "east_m": [-magnitude, magnitude],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(over="ignore", invalid="ignore"):
        features = tracklet_feature_frame(radar)

    np.testing.assert_allclose(features["mean_speed_mps"], [5.0e307])
    np.testing.assert_allclose(features["max_speed_mps"], [5.0e307])


def test_tracklet_features_treat_unrepresentable_duration_as_missing() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [7, 7],
            "frame_index": [0, 1],
            "time_s": [-1.0e308, 1.0e308],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        features = tracklet_feature_frame(radar)

    np.testing.assert_allclose(features["start_time_s"], [-1.0e308])
    np.testing.assert_allclose(features["end_time_s"], [1.0e308])
    assert np.isnan(features.loc[0, "duration_s"])
    assert np.isnan(features.loc[0, "mean_speed_mps"])
    assert np.isnan(features.loc[0, "max_speed_mps"])


def test_tracklet_segmentation_handles_unrepresentable_time_gap() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [7, 7],
            "time_s": [-1.0e308, 1.0e308],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        features = tracklet_feature_frame(radar)

    assert len(features) == 2
    np.testing.assert_allclose(features["duration_s"], [0.0, 0.0])
    np.testing.assert_allclose(features["mean_speed_mps"], [0.0, 0.0])
    np.testing.assert_allclose(features["max_speed_mps"], [0.0, 0.0])


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


def test_frame_context_keeps_large_finite_neighbor_means_finite() -> None:
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
