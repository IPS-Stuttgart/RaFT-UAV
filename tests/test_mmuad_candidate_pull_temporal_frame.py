from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import CandidatePullConfig
from raft_uav.mmuad.candidate_pull import parse_position
from raft_uav.mmuad.candidate_pull import refine_official_results_with_candidate_pull


def test_candidate_pull_uses_one_complete_nearest_candidate_frame() -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA"],
            "Timestamp": [0.05, 0.10, 0.16],
            "Position": ["(0,0,0)", "(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2, 2],
        }
    )
    candidates = pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 0.0, 0.2, 0.2],
            "x_m": [1.0, 2.0, 100.0, 101.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.2, 0.1, 0.9, 0.8],
            "confidence": [0.2, 0.1, 0.9, 0.8],
            "cluster_point_count": [2, 1, 9, 8],
        }
    )

    result = refine_official_results_with_candidate_pull(
        results,
        candidates,
        config=CandidatePullConfig(
            policy="constant",
            smoother="none",
            constant_alpha_xy=1.0,
            constant_alpha_z=1.0,
            top_k=1,
            time_tolerance_s=0.2,
        ),
    )

    positions = [parse_position(value) for value in result.rows["Position"]]
    assert positions[0].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert positions[1].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert positions[2].tolist() == pytest.approx([100.0, 0.0, 0.0])
    assert result.centers["candidate_count"].tolist() == [2, 2, 2]
