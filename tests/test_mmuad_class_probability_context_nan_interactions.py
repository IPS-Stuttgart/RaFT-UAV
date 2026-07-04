from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.schema import CandidateFrame


def test_class_probability_context_zero_fills_nan_interaction_inputs() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "source": ["lidar_360", "lidar_360"],
            "track_id": ["a", "b"],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [0.5, 0.8],
            "cluster_extent_3d_m": [np.nan, 2.0],
        }
    )
    class_probabilities = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "predicted_probability_0": [0.25],
            "predicted_probability_1": [0.25],
            "predicted_probability_2": [0.25],
            "predicted_probability_3": [0.25],
        }
    )

    augmented = attach_class_probability_context(
        CandidateFrame(candidates),
        class_probabilities,
        interaction_columns=("cluster_extent_3d_m",),
    ).rows.sort_values("track_id").reset_index(drop=True)

    interaction = "image_class_prob_0_x_cluster_extent_3d_m"
    assert augmented[interaction].notna().all()
    assert augmented.loc[0, interaction] == pytest.approx(0.0)
    assert augmented.loc[1, interaction] == pytest.approx(0.5)
