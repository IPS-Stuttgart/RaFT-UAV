from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_targets import CandidateOracleTargetConfig
from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets


def test_candidate_oracle_targets_keep_final_finite_duplicate_truth_row() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["stale-truth-match", "final-truth-match"],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.9, 0.1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": ["0", 0.0, "0.000"],
            "x_m": [100.0, 0.0, float("nan")],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )

    target_rows, frame_summary, _summary = build_candidate_oracle_targets(
        candidates,
        truth,
        config=CandidateOracleTargetConfig(score_column="ranker_score"),
    )

    oracle = target_rows.loc[target_rows["candidate_is_oracle"]].iloc[0]
    assert oracle["track_id"] == "final-truth-match"
    assert float(oracle["truth_x_m"]) == pytest.approx(0.0)
    assert int(oracle["candidate_score_rank"]) == 2
    assert frame_summary.loc[0, "oracle_error_3d_m"] == pytest.approx(0.0)
    assert frame_summary.loc[0, "score_top1_error_3d_m"] == pytest.approx(100.0)
