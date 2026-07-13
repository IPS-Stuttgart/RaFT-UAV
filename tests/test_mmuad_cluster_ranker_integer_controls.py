from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.cluster_ranker import _make_sklearn_estimator
from raft_uav.mmuad.cluster_ranker import evaluate_cluster_ranker_loso
from raft_uav.mmuad.cluster_ranker import train_cluster_ranker


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 0.0, 0.0, 0.0],
            "source": ["lidar_360"] * 4,
            "track_id": ["a-good", "a-bad", "b-good", "b-bad"],
            "x_m": [0.0, 10.0, 0.0, 10.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "confidence": [1.0, 1.0, 1.0, 1.0],
            "good_cluster": [True, False, True, False],
            "truth_distance_3d_m": [0.0, 10.0, 0.0, 10.0],
        }
    )


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True, np.nan, np.inf])
def test_cluster_ranker_rejects_lossy_logistic_iteration_counts(
    iterations: object,
) -> None:
    for operation in (train_cluster_ranker, evaluate_cluster_ranker_loso):
        with pytest.raises(ValueError, match="iterations must be a positive integer"):
            operation(_features(), iterations=iterations)


@pytest.mark.parametrize("n_estimators", [0, -1, 1.5, False, np.nan, np.inf])
def test_cluster_ranker_rejects_lossy_estimator_counts(
    n_estimators: object,
) -> None:
    for operation in (train_cluster_ranker, evaluate_cluster_ranker_loso):
        with pytest.raises(ValueError, match="n_estimators must be a positive integer"):
            operation(
                _features(),
                model_type="random-forest-classifier",
                n_estimators=n_estimators,
            )


@pytest.mark.parametrize("minimum", [-1, 1.5, True, np.nan, np.inf])
def test_cluster_ranker_loso_rejects_lossy_minimum_sequence_counts(
    minimum: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="min_train_sequences must be a nonnegative integer",
    ):
        evaluate_cluster_ranker_loso(
            _features(),
            min_train_sequences=minimum,
        )


def test_cluster_ranker_accepts_integer_equivalent_controls() -> None:
    model = train_cluster_ranker(_features(), iterations=2.0)
    predictions, folds, pooled = evaluate_cluster_ranker_loso(
        _features(),
        iterations=2.0,
        min_train_sequences=1.0,
    )

    assert model.model_type == "logistic"
    assert len(predictions) == 4
    assert len(folds) == 2
    assert pooled.loc[0, "fold_count"] == 2


def test_cluster_ranker_estimator_accepts_integer_equivalent_count() -> None:
    pytest.importorskip("sklearn")

    estimator, score_transform = _make_sklearn_estimator(
        model_type="random-forest-classifier",
        random_state=13,
        n_estimators=2.0,
    )

    assert estimator.n_estimators == 2
    assert score_transform == "probability"
