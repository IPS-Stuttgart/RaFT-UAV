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
    "invalid_pair",
    [
        (0.0, 0.0),
        (-1.0, -2.0),
        (np.nan, np.nan),
    ],
)
def test_sequence_probability_relabel_rejects_empty_probability_mass(
    invalid_pair: tuple[float, float],
) -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [invalid_pair[0], 0.25],
            "predicted_probability_1": [invalid_pair[1], 0.75],
        }
    )

    with pytest.raises(
        ValueError,
        match=(
            "sequence prediction probabilities have no positive finite mass for "
            "sequence\\(s\\): 'seq0001'"
        ),
    ):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )


def test_sequence_probability_relabel_keeps_valid_unnormalized_probabilities() -> None:
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
