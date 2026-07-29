from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows(
    primary_scores: list[object],
    confidence_scores: list[object],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["malformed", "finite"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": primary_scores,
            "confidence": confidence_scores,
        }
    )


@pytest.mark.parametrize("invalid_primary", [1.0 + 2.0j, True])
def test_diversity_uses_fallback_for_malformed_primary_scores(
    invalid_primary: object,
) -> None:
    rows = _rows([invalid_primary, 0.5], [0.1, 0.2])

    output = diversify_candidate_reservoir(rows, radius_m=1.0)

    assert output["track_id"].tolist() == ["finite"]
    assert output["candidate_reservoir_score"].tolist() == pytest.approx([0.5])


def test_diversity_rejects_malformed_confidence_fallback() -> None:
    rows = _rows([np.nan, 0.5], [1.0 + 2.0j, 0.1])

    output = diversify_candidate_reservoir(rows, radius_m=1.0)

    assert output["track_id"].tolist() == ["finite"]
    assert output["candidate_reservoir_score"].tolist() == pytest.approx([0.5])


def test_diversity_preserves_real_scores_in_complex_upcast_columns() -> None:
    rows = _rows([0.8, 0.7 + 2.0j], [0.1, 0.6])
    assert np.iscomplexobj(rows["candidate_reservoir_score"].to_numpy())

    output = diversify_candidate_reservoir(rows, radius_m=1.0)

    assert output["track_id"].tolist() == ["malformed"]
    assert output["candidate_reservoir_score"].tolist() == pytest.approx([0.8])
    assert np.issubdtype(output["candidate_reservoir_score"].dtype, np.floating)
