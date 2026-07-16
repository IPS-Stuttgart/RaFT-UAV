from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.sequence_classifier_fusion as fusion
from raft_uav.mmuad.sequence_classifier_fusion import fuse_sequence_probabilities
from raft_uav.mmuad.sequence_classifier_fusion import select_train_safe_fusion


def _probabilities(first: float, second: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "predicted_probability_0": [first],
            "predicted_probability_1": [second],
            "predicted_probability_2": [0.0],
            "predicted_probability_3": [0.0],
        }
    )


@pytest.mark.parametrize(
    "weight",
    [
        -0.01,
        1.01,
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        0.5 + 0.0j,
        np.array([0.5]),
        np.ma.masked,
    ],
)
def test_fusion_rejects_invalid_image_weights(weight: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"image_weight must be a finite real scalar in \[0, 1\]",
    ):
        fuse_sequence_probabilities(
            _probabilities(1.0, 0.0),
            _probabilities(0.0, 1.0),
            image_weight=weight,
        )


def test_fusion_accepts_zero_dimensional_real_weight() -> None:
    fused = fuse_sequence_probabilities(
        _probabilities(1.0, 0.0),
        _probabilities(0.0, 1.0),
        image_weight=np.array(0.25),
    )

    assert fused["predicted_probability_0"].tolist() == pytest.approx([0.25])
    assert fused["predicted_probability_1"].tolist() == pytest.approx([0.75])
    assert fused["class_source"].tolist() == [
        "sequence-fused-image-weight-0.25"
    ]


def test_selection_rejects_invalid_grid_before_training(monkeypatch) -> None:
    def unexpected_legacy_call(**_kwargs):
        raise AssertionError("legacy selection should not run")

    monkeypatch.setattr(
        fusion,
        "_LEGACY_SELECT_TRAIN_SAFE_FUSION",
        unexpected_legacy_call,
    )

    with pytest.raises(
        ValueError,
        match=r"image_weights\[1\] must be a finite real scalar in \[0, 1\]",
    ):
        select_train_safe_fusion(
            image_train_features=pd.DataFrame(),
            nonimage_train_features=pd.DataFrame(),
            image_predict_features=pd.DataFrame(),
            nonimage_predict_features=pd.DataFrame(),
            train_labels={},
            model_specs=[],
            image_weights=[0.5, 2.0],
        )
