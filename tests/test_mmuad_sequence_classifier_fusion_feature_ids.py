from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.sequence_classifier_fusion import FusionModelSpec
from raft_uav.mmuad.sequence_classifier_fusion import _sequence_indexed
from raft_uav.mmuad.sequence_classifier_fusion import select_train_safe_fusion


@pytest.mark.parametrize(
    "sequence_ids",
    [
        ["seq", "seq"],
        [1, "1"],
    ],
)
def test_sequence_feature_index_rejects_duplicate_normalized_ids(
    sequence_ids: list[object],
) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "feature": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="image_train_features contains duplicate sequence_id values: '1'|"
        "image_train_features contains duplicate sequence_id values: 'seq'",
    ):
        _sequence_indexed(rows, "image_train_features")


@pytest.mark.parametrize("sequence_id", [None, np.nan, pd.NA, "", "   "])
def test_sequence_feature_index_rejects_missing_ids(sequence_id: object) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "feature": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="predict_features contains missing sequence_id values",
    ):
        _sequence_indexed(rows, "predict_features")


def test_train_safe_fusion_rejects_duplicate_feature_rows_before_training() -> None:
    duplicate_image_train = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "feature": [1.0, 2.0],
        }
    )
    one_row = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "feature": [1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="image_train_features contains duplicate sequence_id values: 'seq'",
    ):
        select_train_safe_fusion(
            image_train_features=duplicate_image_train,
            nonimage_train_features=one_row,
            image_predict_features=one_row,
            nonimage_predict_features=one_row,
            train_labels={"seq": "0"},
            model_specs=[
                FusionModelSpec(
                    method="random-forest",
                    n_estimators=1,
                    max_depth=None,
                    random_state=0,
                )
            ],
            image_weights=[0.5],
            cv_folds=2,
        )


def test_sequence_feature_index_preserves_opaque_zero_padded_ids() -> None:
    indexed = _sequence_indexed(
        pd.DataFrame(
            {
                "sequence_id": ["001", "010"],
                "feature": [1.0, 2.0],
            }
        ),
        "image_train_features",
    )

    assert indexed.index.tolist() == ["001", "010"]
