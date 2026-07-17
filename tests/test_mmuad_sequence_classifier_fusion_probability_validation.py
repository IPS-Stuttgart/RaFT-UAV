from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.sequence_classifier_fusion import (
    OFFICIAL_SEQUENCE_CLASS_LABELS,
    fuse_sequence_probabilities,
)


def _probabilities(
    sequence_ids: list[object],
    *,
    first_probability: object = 1.0,
) -> pd.DataFrame:
    data: dict[str, list[object]] = {"sequence_id": sequence_ids}
    for index, label in enumerate(OFFICIAL_SEQUENCE_CLASS_LABELS):
        data[f"predicted_probability_{label}"] = [
            first_probability if index == 0 else 0.0
            for _ in sequence_ids
        ]
    return pd.DataFrame(data)


def test_fusion_rejects_duplicate_sequence_probability_rows() -> None:
    image = _probabilities(["seq", "seq"])
    nonimage = _probabilities(["seq"])

    with pytest.raises(
        ValueError,
        match="image_probabilities contains duplicate sequence_id values: 'seq'",
    ):
        fuse_sequence_probabilities(
            image,
            nonimage,
            image_weight=0.5,
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        np.nan,
        np.inf,
        True,
        0.5 + 0.0j,
        np.ma.masked,
    ],
)
def test_fusion_rejects_invalid_probability_values(value: object) -> None:
    image = _probabilities(["seq"], first_probability=value)
    nonimage = _probabilities(["seq"])

    with pytest.raises(
        ValueError,
        match="image_probabilities must contain finite non-negative real probabilities",
    ):
        fuse_sequence_probabilities(
            image,
            nonimage,
            image_weight=0.5,
        )


def test_fusion_rejects_zero_fused_probability_mass() -> None:
    image = _probabilities(["seq"], first_probability=0.0)
    nonimage = _probabilities(["seq"], first_probability=0.0)

    with pytest.raises(
        ValueError,
        match="fused probabilities contain zero probability mass.*'seq'",
    ):
        fuse_sequence_probabilities(
            image,
            nonimage,
            image_weight=0.5,
        )


def test_fusion_uses_valid_modality_when_other_row_has_zero_mass() -> None:
    image = _probabilities(["seq"], first_probability=0.0)
    nonimage = _probabilities(["seq"])

    fused = fuse_sequence_probabilities(
        image,
        nonimage,
        image_weight=0.5,
    )

    first_label = str(OFFICIAL_SEQUENCE_CLASS_LABELS[0])
    assert fused["predicted_class"].astype(str).tolist() == [first_label]
    assert fused[f"predicted_probability_{first_label}"].tolist() == pytest.approx([1.0])


def test_fusion_accepts_an_empty_modality_table() -> None:
    nonimage = _probabilities(["seq"])

    fused = fuse_sequence_probabilities(
        pd.DataFrame(),
        nonimage,
        image_weight=0.5,
    )

    first_label = str(OFFICIAL_SEQUENCE_CLASS_LABELS[0])
    assert fused["sequence_id"].tolist() == ["seq"]
    assert fused["predicted_class"].astype(str).tolist() == [first_label]
