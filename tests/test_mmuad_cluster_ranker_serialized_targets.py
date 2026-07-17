from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.cluster_ranker import (
    _binary_auc,
    _ranker_prediction_summary,
    predict_cluster_scores,
    train_cluster_ranker,
)


def _features(labels: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["lidar"] * 4,
            "confidence": [0.95, 0.05, 0.85, 0.15],
            "good_cluster": labels,
        }
    )


def test_cluster_ranker_parses_serialized_binary_targets() -> None:
    features = _features(["True", "False", "yes", "0"])

    model = train_cluster_ranker(features)
    scores = predict_cluster_scores(features, model)

    assert model.constant_score is None
    assert scores[[0, 2]].mean() > scores[[1, 3]].mean()


def test_cluster_ranker_rejects_ambiguous_serialized_target() -> None:
    with pytest.raises(ValueError, match="good_cluster.*maybe"):
        train_cluster_ranker(_features(["True", "False", "maybe", "0"]))


def test_cluster_ranker_auc_parses_serialized_binary_targets() -> None:
    auc = _binary_auc(
        pd.Series([0.1, 0.9]),
        pd.Series(["False", "True"]),
    )

    assert auc == pytest.approx(1.0)


def test_cluster_ranker_summary_parses_serialized_binary_targets() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 1.0],
            "source": ["lidar", "lidar"],
            "track_id": ["positive", "negative"],
            "truth_distance_3d_m": [1.0, 10.0],
            "ranker_score": [0.9, 0.1],
            "good_cluster": ["True", "False"],
        }
    )

    summary = _ranker_prediction_summary(
        rows,
        sequence="seq",
        split="heldout_sequence",
        protocol="test",
    )

    assert summary["positive_candidate_rows"] == 1
    assert summary["positive_candidate_rate"] == pytest.approx(0.5)
    assert summary["score_auc"] == pytest.approx(1.0)
