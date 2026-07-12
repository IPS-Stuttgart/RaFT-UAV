from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import CandidatePullConfig
from raft_uav.mmuad.candidate_pull import parse_position
from raft_uav.mmuad.candidate_pull import refine_official_results_with_candidate_pull


def test_candidate_pull_normalizes_nondefault_result_index() -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(2,0,0)"],
            "Classification": [2, 2],
        },
        index=[10, 20],
    )
    candidates = pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "x_m": [10.0, 4.0],
            "y_m": [2.0, 4.0],
            "z_m": [6.0, 8.0],
            "ranker_score": [0.9, 0.8],
            "confidence": [0.8, 0.8],
            "cluster_point_count": [20, 20],
        }
    )

    result = refine_official_results_with_candidate_pull(
        results,
        candidates,
        config=CandidatePullConfig(
            policy="constant",
            smoother="none",
            constant_alpha_xy=0.5,
            constant_alpha_z=0.25,
            top_k=1,
        ),
    )

    assert result.rows.index.tolist() == [0, 1]
    assert parse_position(result.rows.iloc[0]["Position"]).tolist() == pytest.approx(
        [5.0, 1.0, 1.5]
    )
    assert parse_position(result.rows.iloc[1]["Position"]).tolist() == pytest.approx(
        [3.0, 2.0, 2.0]
    )
    assert result.provenance["matched_candidate_center_count"] == 2
