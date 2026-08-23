from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context


def _candidate_rows(sequence_id: object) -> pd.DataFrame:
    return pd.DataFrame(
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


def _probability_rows(sequence_id: object = "001") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "predicted_class": [2],
        }
    )


def test_candidate_sequence_ids_are_stripped_before_probability_join() -> None:
    row = attach_class_probability_context(
        _candidate_rows(" 001 "),
        _probability_rows(),
        fill_missing="error",
        interaction_columns=(),
    ).rows.iloc[0]

    assert row["sequence_id"] == "001"
    assert row["image_class_probability_available"] == pytest.approx(1.0)
    assert row["image_class_prob_2"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "sequence_id",
    [None, pd.NA, np.nan, "", "   ", "nan", "None", "<NA>"],
)
def test_candidate_rows_reject_missing_sequence_ids(sequence_id: object) -> None:
    with pytest.raises(
        ValueError,
        match="candidate sequence identifiers must be non-empty",
    ):
        attach_class_probability_context(
            _candidate_rows(sequence_id),
            _probability_rows(),
            interaction_columns=(),
        )


def test_candidate_rows_accept_equivalent_sequence_aliases() -> None:
    candidates = _candidate_rows("001")
    candidates["Sequence"] = [" 001 "]

    row = attach_class_probability_context(
        candidates,
        _probability_rows(),
        fill_missing="error",
        interaction_columns=(),
    ).rows.iloc[0]

    assert row["sequence_id"] == "001"
    assert row["image_class_prob_2"] == pytest.approx(1.0)


def test_candidate_rows_reject_conflicting_sequence_aliases() -> None:
    candidates = _candidate_rows("001")
    candidates["Sequence"] = ["002"]

    with pytest.raises(
        ValueError,
        match="candidate table has conflicting sequence identifier columns",
    ):
        attach_class_probability_context(
            candidates,
            _probability_rows(),
            interaction_columns=(),
        )
