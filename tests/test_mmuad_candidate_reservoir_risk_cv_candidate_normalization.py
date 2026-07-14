from __future__ import annotations

import pandas as pd

import raft_uav.mmuad.candidate_reservoir_risk_cv as risk_cv


def test_direct_risk_cv_normalizes_candidates_before_sequence_matching() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": [" seqA ", "seqB"],
            "time_s": [0.0, 0.0],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 1.0],
            "z_m": [0.0, 1.0],
        }
    )

    candidate_rows, _, sequences = risk_cv._validated_inputs(candidates, truth)

    assert sequences == ["seqA", "seqB"]
    assert candidate_rows["sequence_id"].tolist() == ["seqA", "seqB"]
    assert {"x_m", "y_m", "z_m"}.issubset(candidate_rows.columns)
