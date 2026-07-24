from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import (
    relabel_track5_classification_from_sequence_predictions,
)


def _pose_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,1)", "(1,0,1)", "(5,0,2)"],
            "Classification": [0, 0, 3],
        }
    )


@pytest.mark.parametrize(
    "invalid_value",
    ["not-a-number", np.nan, np.inf, -0.25],
)
def test_sequence_probability_relabel_rejects_invalid_probability_cells(
    invalid_value: object,
) -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [invalid_value, 0.25],
            "predicted_probability_1": [1.0, 0.75],
        }
    )

    with pytest.raises(
        ValueError,
        match="non-finite, non-numeric, or negative values",
    ):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )


def test_sequence_probability_relabel_rejects_zero_probability_mass() -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [0.0, 0.25],
            "predicted_probability_1": [0.0, 0.75],
        }
    )

    with pytest.raises(
        ValueError,
        match="no positive mass for sequence\\(s\\): 'seq0001'",
    ):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )


def test_sequence_probability_relabel_accepts_unnormalized_positive_mass() -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [2.0, 7.0],
            "predicted_probability_1": [8.0, 3.0],
        }
    )

    result = relabel_track5_classification_from_sequence_predictions(
        _pose_rows(),
        predictions,
    )

    assert result.rows["Classification"].tolist() == [1, 1, 0]
    assert result.diagnostics["source_classification_probability"].tolist() == pytest.approx(
        [0.8, 0.8, 0.7]
    )
