from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf, "not-a-score"])
def test_diversity_uses_rowwise_finite_confidence_fallback(
    invalid_score: object,
) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["fallback-best", "finite-primary"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [invalid_score, 0.5],
            "confidence": [0.9, 0.1],
        }
    )

    output = diversify_candidate_reservoir(rows, radius_m=1.0)

    assert output["track_id"].tolist() == ["fallback-best"]
    assert output["candidate_reservoir_score"].tolist() == pytest.approx([0.9])


def test_diversity_keeps_finite_primary_score_over_confidence() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["primary-best", "confidence-best"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [0.8, 0.7],
            "confidence": [0.1, 0.99],
        }
    )

    output = diversify_candidate_reservoir(rows, radius_m=1.0)

    assert output["track_id"].tolist() == ["primary-best"]
    assert output["candidate_reservoir_score"].tolist() == pytest.approx([0.8])
