from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.cluster_ranker import (
    _standardize_training_matrix,
    label_cluster_features_against_truth,
    train_cluster_ranker,
)


def test_cluster_ranker_truth_norm_stays_finite_for_large_finite_residual() -> None:
    features = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [1.0e308],
            "y_m": [1.0e308],
            "z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with np.errstate(all="raise"):
        labeled = label_cluster_features_against_truth(features, truth)

    distance = float(labeled.loc[0, "truth_distance_3d_m"])
    assert math.isfinite(distance)
    assert distance / 1.0e308 == pytest.approx(math.sqrt(2.0))
    assert labeled.loc[0, "truth_matched"]


def test_cluster_ranker_truth_norm_preserves_tiny_finite_residual() -> None:
    features = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [1.0e-308],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with np.errstate(all="raise"):
        labeled = label_cluster_features_against_truth(features, truth)

    distance = float(labeled.loc[0, "truth_distance_3d_m"])
    assert distance > 0.0
    assert distance / 1.0e-308 == pytest.approx(1.0)


def test_cluster_ranker_standardization_keeps_large_finite_scale() -> None:
    matrix = np.array(
        [
            [1.0e308, 1.0e308],
            [-1.0e308, 1.0e308],
            [np.nan, np.nan],
        ]
    )

    with np.errstate(all="raise"):
        filled, means, scales = _standardize_training_matrix(matrix)

    assert np.isfinite(filled).all()
    assert np.isfinite(means).all()
    assert np.isfinite(scales).all()
    assert means[0] == 0.0
    assert means[1] / 1.0e308 == pytest.approx(1.0)
    assert scales[0] / 1.0e308 == pytest.approx(1.0)
    assert scales[1] == 1.0


def test_cluster_ranker_training_handles_large_finite_feature_values() -> None:
    features = pd.DataFrame(
        {
            "source": ["lidar_360", "lidar_360"],
            "x_m": [1.0e308, -1.0e308],
            "good_cluster": [True, False],
        }
    )

    with np.errstate(all="raise"):
        model = train_cluster_ranker(
            features,
            iterations=5,
            learning_rate=0.1,
        )

    assert all(math.isfinite(value) for value in model.feature_means)
    assert all(math.isfinite(value) for value in model.feature_scales)
    assert all(math.isfinite(value) for value in model.weights)
    assert math.isfinite(model.bias)
