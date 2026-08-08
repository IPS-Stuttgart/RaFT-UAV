from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.sequence_classifier_fusion import (
    OFFICIAL_SEQUENCE_CLASS_LABELS,
    _sequence_indexed,
    fuse_sequence_probabilities,
)


def _one_hot_probabilities(sequence_id: str, class_index: int) -> pd.DataFrame:
    row: dict[str, object] = {"sequence_id": sequence_id}
    for index, label in enumerate(OFFICIAL_SEQUENCE_CLASS_LABELS):
        row[f"predicted_probability_{label}"] = 1.0 if index == class_index else 0.0
    return pd.DataFrame([row])


def test_sequence_feature_index_trims_surrounding_whitespace() -> None:
    indexed = _sequence_indexed(
        pd.DataFrame(
            {
                "sequence_id": [" seq-a ", "001"],
                "feature": [1.0, 2.0],
            }
        ),
        "image_train_features",
    )

    assert indexed.index.tolist() == ["seq-a", "001"]


def test_sequence_feature_index_rejects_duplicates_after_trimming() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq-a", " seq-a "],
            "feature": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="image_train_features contains duplicate sequence_id values: 'seq-a'",
    ):
        _sequence_indexed(rows, "image_train_features")


def test_probability_fusion_joins_whitespace_equivalent_sequences() -> None:
    image = _one_hot_probabilities(" seq-a ", 0)
    nonimage = _one_hot_probabilities("seq-a", 1)

    fused = fuse_sequence_probabilities(
        image,
        nonimage,
        image_weight=0.5,
    )

    class_zero = f"predicted_probability_{OFFICIAL_SEQUENCE_CLASS_LABELS[0]}"
    class_one = f"predicted_probability_{OFFICIAL_SEQUENCE_CLASS_LABELS[1]}"
    assert fused["sequence_id"].tolist() == ["seq-a"]
    assert fused[class_zero].tolist() == pytest.approx([0.5])
    assert fused[class_one].tolist() == pytest.approx([0.5])
