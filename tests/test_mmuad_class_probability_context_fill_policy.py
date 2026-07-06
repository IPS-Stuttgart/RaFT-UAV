from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.schema import CandidateFrame


def test_class_probability_context_rejects_unknown_fill_policy() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "lidar_360"],
            "track_id": ["a", "b"],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "predicted_probability_0": [1.0],
            "predicted_probability_1": [0.0],
            "predicted_probability_2": [0.0],
            "predicted_probability_3": [0.0],
        }
    )

    with pytest.raises(ValueError, match="fill_missing"):
        attach_class_probability_context(
            CandidateFrame(candidates),
            probabilities,
            fill_missing="zreo",
            interaction_columns=(),
        )
