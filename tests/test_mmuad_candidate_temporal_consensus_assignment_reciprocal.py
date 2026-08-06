from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_temporal_consensus import TemporalConsensusConfig
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
)
from raft_uav.mmuad.schema import CandidateFrame


def test_one_to_one_assignment_is_reciprocal_under_tied_costs() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["early_far", "early_near", "late_a", "late_b"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "track_id": ["early_far", "early_near", "late_a", "late_b"],
            "x_m": [0.0, 1.0, 2.0, 2.0],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "confidence": [0.5] * 4,
            "ranker_score": [0.5] * 4,
        }
    )
    assigned = add_assignment_temporal_candidate_consensus(
        CandidateFrame(candidates),
        config=TemporalConsensusConfig(
            max_time_gap_s=1.1,
            max_speed_mps=10.0,
            distance_scale_m=2.0,
            acceleration_scale_mps2=5.0,
        ),
        assignment_mode="one-to-one",
    ).rows.set_index("track_id")

    forward_ids = {
        str(assigned.loc[track_id, "candidate_temporal_forward_track_id"])
        for track_id in ("early_far", "early_near")
    }
    assert forward_ids == {"late_a", "late_b"}
    for early_track_id in ("early_far", "early_near"):
        late_track_id = str(
            assigned.loc[early_track_id, "candidate_temporal_forward_track_id"]
        )
        assert (
            assigned.loc[late_track_id, "candidate_temporal_backward_track_id"]
            == early_track_id
        )
