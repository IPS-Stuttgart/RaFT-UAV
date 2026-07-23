from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import _normalize_frame
from raft_uav.mmuad.track5_classification_relabel import (
    _sequence_prediction_labels,
)
from raft_uav.mmuad.track5_classification_relabel import (
    relabel_track5_classification,
)
from raft_uav.mmuad.track5_classification_relabel import (
    relabel_track5_classification_from_sequence_predictions,
)


def _official_rows(sequence: object = "seq0001") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": [sequence],
            "Timestamp": [0.0],
            "Position": ["(0,0,1)"],
            "Classification": [0],
        }
    )


def test_relabel_rejects_missing_pose_sequence_before_string_coercion() -> None:
    with pytest.raises(
        ValueError,
        match="pose_submission contains missing or blank Sequence identifiers",
    ):
        relabel_track5_classification(
            _official_rows(np.nan),
            _official_rows(np.nan),
        )


def test_relabel_rejects_blank_classification_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="classification_submission contains missing or blank Sequence identifiers",
    ):
        relabel_track5_classification(
            _official_rows(),
            _official_rows("   "),
        )


def test_relabel_rejects_missing_sequence_prediction_ids() -> None:
    predictions = pd.DataFrame(
        {
            "heldout_sequence": ["seq0001", None],
            "predicted_probability_0": [0.1, 0.2],
            "predicted_probability_1": [0.9, 0.8],
        }
    )

    with pytest.raises(
        ValueError,
        match="sequence prediction table contains missing or blank Sequence identifiers",
    ):
        relabel_track5_classification_from_sequence_predictions(
            _official_rows(),
            predictions,
        )


def test_relabel_preserves_literal_nan_sequence_identifier() -> None:
    result = relabel_track5_classification(
        _official_rows("nan"),
        _official_rows("nan").assign(Classification=2),
    )

    assert result.rows["Sequence"].tolist() == ["nan"]
    assert result.rows["Classification"].tolist() == [2]


def test_public_relabel_functions_use_sequence_validation_hooks() -> None:
    assert relabel_track5_classification.__globals__["_normalize_frame"] is _normalize_frame
    assert (
        relabel_track5_classification_from_sequence_predictions.__globals__[
            "_sequence_prediction_labels"
        ]
        is _sequence_prediction_labels
    )
