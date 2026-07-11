from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.schema import CandidateFrame


def test_class_probability_context_falls_back_per_row_to_predicted_class() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqB"],
                "time_s": [0.0, 0.0],
                "source": ["lidar_360", "lidar_360"],
                "track_id": ["a", "b"],
                "x_m": [1.0, 2.0],
                "y_m": [0.0, 0.0],
                "z_m": [1.0, 1.0],
                "confidence": [0.5, 0.6],
            }
        )
    )
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "predicted_probability_0": [0.1, np.nan],
            "predicted_probability_1": [0.2, np.nan],
            "predicted_probability_2": [0.6, np.nan],
            "predicted_probability_3": [0.1, np.nan],
            "predicted_class": [2, 3],
        }
    )

    rows = attach_class_probability_context(
        candidates,
        probabilities,
        fill_missing="error",
        interaction_columns=(),
    ).rows.sort_values("sequence_id").reset_index(drop=True)

    seq_a = rows.loc[rows["sequence_id"].eq("seqA")].iloc[0]
    seq_b = rows.loc[rows["sequence_id"].eq("seqB")].iloc[0]
    assert seq_a["image_class_prob_2"] == pytest.approx(0.6)
    assert seq_b["image_class_probability_available"] == pytest.approx(1.0)
    assert seq_b["image_class_prob_0"] == pytest.approx(0.0)
    assert seq_b["image_class_prob_3"] == pytest.approx(1.0)
    assert seq_b["image_predicted_class_id"] == pytest.approx(3.0)
