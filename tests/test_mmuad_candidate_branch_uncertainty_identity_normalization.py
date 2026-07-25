from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_branch_uncertainty import (
    attach_branch_uncertainty_context,
)


def test_branch_context_normalizes_source_and_branch_group_identities() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": [" RF ", "rf"],
            "candidate_branch": [" RAW ", "raw"],
            "track_id": ["higher", "lower"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.9, 0.5],
        }
    )

    contextual = attach_branch_uncertainty_context(candidates).rows.set_index(
        "track_id"
    )

    assert contextual["candidate_reservoir_frame_branch_count"].tolist() == [1.0, 1.0]
    assert contextual["candidate_reservoir_branch_candidate_count"].tolist() == [2.0, 2.0]
    assert contextual["candidate_reservoir_source_branch_candidate_count"].tolist() == [
        2.0,
        2.0,
    ]
    assert contextual["candidate_reservoir_branch_score_rank"].tolist() == [1.0, 2.0]
    assert contextual["candidate_reservoir_source_branch_score_rank"].tolist() == [
        1.0,
        2.0,
    ]
    assert contextual["candidate_reservoir_branch_score_gap"].tolist() == [0.0, 0.4]
    assert contextual["candidate_reservoir_source_branch_score_gap"].tolist() == [
        0.0,
        0.4,
    ]
