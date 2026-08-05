from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_temporal_consensus import TemporalConsensusConfig
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
)
from raft_uav.mmuad.schema import CandidateFrame


def test_one_to_one_assignment_maximizes_valid_match_cardinality() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["source_a", "source_b", "source_c", "source_d"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "track_id": [
                "previous_origin",
                "previous_right",
                "current_origin",
                "current_left",
            ],
            "x_m": [0.0, 10.0, 0.0, -9.9],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "confidence": [0.5] * 4,
            "ranker_score": [0.5] * 4,
        }
    )
    config = TemporalConsensusConfig(
        max_time_gap_s=1.1,
        max_speed_mps=10.0,
        distance_scale_m=2.0,
        acceleration_scale_mps2=5.0,
    )

    assigned = add_assignment_temporal_candidate_consensus(
        CandidateFrame(candidates),
        config=config,
        assignment_mode="one-to-one",
    ).rows

    previous = assigned.loc[assigned["time_s"] == 0.0].set_index("track_id")
    current = assigned.loc[assigned["time_s"] == 1.0].set_index("track_id")

    assert (
        previous["candidate_reservoir_temporal_forward_assignment_matched"].sum()
        == 2.0
    )
    assert (
        current["candidate_reservoir_temporal_backward_assignment_matched"].sum()
        == 2.0
    )
    assert (
        current.loc["current_origin", "candidate_temporal_backward_track_id"]
        == "previous_right"
    )
    assert (
        current.loc["current_left", "candidate_temporal_backward_track_id"]
        == "previous_origin"
    )
