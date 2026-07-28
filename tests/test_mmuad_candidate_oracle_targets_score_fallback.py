from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_targets import CandidateOracleTargetConfig
from raft_uav.mmuad.candidate_oracle_targets import _candidate_score
from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets


@pytest.mark.parametrize("invalid_primary", [np.inf, -np.inf, True, 1.0 + 2.0j])
def test_candidate_score_uses_fallback_for_invalid_primary_values(
    invalid_primary: object,
) -> None:
    rows = pd.DataFrame(
        {
            "primary_score": [invalid_primary, 0.5],
            "fallback_score": [0.75, 0.9],
        }
    )
    config = CandidateOracleTargetConfig(
        score_column="primary_score",
        fallback_score_columns=("fallback_score",),
    )

    scores = _candidate_score(rows, config=config)

    assert scores.tolist() == pytest.approx([0.75, 0.5])


def test_nonfinite_primary_score_does_not_corrupt_top1_oracle_diagnostic() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["sequence", "sequence"],
            "time_s": [0.0, 0.0],
            "track_id": ["malformed-high-score", "valid-top-score"],
            "x_m": [10.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "primary_score": [np.inf, 0.2],
            "fallback_score": [0.1, 0.9],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["sequence"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )
    config = CandidateOracleTargetConfig(
        score_column="primary_score",
        fallback_score_columns=("fallback_score",),
    )

    targets, frames, _ = build_candidate_oracle_targets(
        candidates,
        truth,
        config=config,
    )

    valid = targets.loc[targets["track_id"] == "valid-top-score"].iloc[0]
    assert int(valid["candidate_score_rank"]) == 1
    assert frames.iloc[0]["score_top1_error_3d_m"] == pytest.approx(0.0)
