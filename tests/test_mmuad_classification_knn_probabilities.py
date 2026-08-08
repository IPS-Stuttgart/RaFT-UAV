from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.classification_cli import main as sequence_classifier_main


def test_loso_k_nearest_neighbor_writes_weighted_vote_probabilities(tmp_path) -> None:
    feature_table = tmp_path / "sequence_features.csv"
    reference = tmp_path / "sequence_labels.csv"
    predictions_csv = tmp_path / "loso_predictions.csv"
    pd.DataFrame(
        {
            "sequence_id": ["target", "near_1", "near_0a", "near_0b", "far_1"],
            "signal": [0.0, 0.20, 0.21, 0.22, 10.0],
        }
    ).to_csv(feature_table, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["target", "near_1", "near_0a", "near_0b", "far_1"],
            "uav_type": [0, 1, 0, 0, 1],
        }
    ).to_csv(reference, index=False)

    assert (
        sequence_classifier_main(
            [
                "--loso-eval",
                "--method",
                "nearest-neighbor",
                "--k",
                "3",
                "--reference",
                str(reference),
                "--selected-tracklets",
                str(feature_table),
                "--loso-predictions-csv",
                str(predictions_csv),
            ]
        )
        == 0
    )

    predictions = pd.read_csv(predictions_csv)
    target = predictions.loc[
        predictions["heldout_sequence"].astype(str).eq("target")
    ].iloc[0]
    class_0_weight = 1.0 / 0.21 + 1.0 / 0.22
    class_1_weight = 1.0 / 0.20
    total = class_0_weight + class_1_weight

    assert str(target["predicted_class"]) == "0"
    assert target["predicted_probability_0"] == pytest.approx(class_0_weight / total)
    assert target["predicted_probability_1"] == pytest.approx(class_1_weight / total)
    assert target["predicted_probability_2"] == 0.0
    assert target["predicted_probability_3"] == 0.0
    probability_columns = [
        "predicted_probability_0",
        "predicted_probability_1",
        "predicted_probability_2",
        "predicted_probability_3",
    ]
    assert float(target[probability_columns].sum()) == pytest.approx(1.0)
    assert 0.0 < float(target["predicted_probability_1"]) < 1.0
