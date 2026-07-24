from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_frame(sequence_id: str = "001") -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": [sequence_id],
                "time_s": [0.0],
                "source": ["lidar_360"],
                "track_id": ["candidate-1"],
                "x_m": [1.0],
                "y_m": [2.0],
                "z_m": [3.0],
            }
        )
    )


def test_in_memory_probability_headers_and_sequence_ids_are_stripped() -> None:
    probabilities = pd.DataFrame(
        {
            " Sequence ": [" 001 "],
            " predicted_probability_0 ": [0.1],
            " predicted_probability_1 ": [0.2],
            " predicted_probability_2 ": [0.6],
            " predicted_probability_3 ": [0.1],
        }
    )

    augmented = attach_class_probability_context(
        _candidate_frame(),
        probabilities,
        fill_missing="error",
        interaction_columns=(),
    )

    row = augmented.rows.iloc[0]
    assert row["image_class_probability_available"] == pytest.approx(1.0)
    assert row["image_class_prob_2"] == pytest.approx(0.6)
    assert row["image_predicted_class_id"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "sequence_id",
    [None, pd.NA, np.nan, "", "   ", "nan", "None", "<NA>"],
)
def test_in_memory_probability_rows_reject_missing_sequence_ids(
    sequence_id: object,
) -> None:
    probabilities = pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "predicted_class": [2],
        }
    )

    with pytest.raises(
        ValueError,
        match="class probability sequence identifiers must be non-empty",
    ):
        attach_class_probability_context(
            _candidate_frame(),
            probabilities,
            interaction_columns=(),
        )


def test_in_memory_probability_rows_accept_equivalent_sequence_aliases() -> None:
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["001"],
            "Sequence": [" 001 "],
            "predicted_class": [2],
        }
    )

    row = attach_class_probability_context(
        _candidate_frame(),
        probabilities,
        fill_missing="error",
        interaction_columns=(),
    ).rows.iloc[0]

    assert row["image_class_prob_2"] == pytest.approx(1.0)


def test_in_memory_probability_rows_reject_conflicting_sequence_aliases() -> None:
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["001"],
            "Sequence": ["002"],
            "predicted_class": [2],
        }
    )

    with pytest.raises(
        ValueError,
        match="conflicting sequence identifier columns",
    ):
        attach_class_probability_context(
            _candidate_frame(),
            probabilities,
            interaction_columns=(),
        )
