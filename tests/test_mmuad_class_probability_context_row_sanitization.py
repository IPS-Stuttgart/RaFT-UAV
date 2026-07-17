from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["candidate"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "confidence": [0.5],
        }
    )


@pytest.mark.parametrize("invalid_probability", [-1.0, np.nan, np.inf, -np.inf])
def test_class_probability_context_sanitizes_rows_before_duplicate_averaging(
    invalid_probability: float,
) -> None:
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "predicted_probability_0": [invalid_probability, 2.0],
            "predicted_probability_1": [1.0, 0.0],
            "predicted_probability_2": [0.0, 0.0],
            "predicted_probability_3": [0.0, 0.0],
        }
    )

    row = attach_class_probability_context(
        CandidateFrame(_candidate_rows()),
        probabilities,
        interaction_columns=(),
    ).rows.iloc[0]

    assert row["image_class_prob_0"] == pytest.approx(2.0 / 3.0)
    assert row["image_class_prob_1"] == pytest.approx(1.0 / 3.0)
    assert row["image_class_prob_2"] == pytest.approx(0.0)
    assert row["image_class_prob_3"] == pytest.approx(0.0)
