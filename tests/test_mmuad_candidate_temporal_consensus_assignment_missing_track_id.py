from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import TemporalConsensusConfig
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
)


def test_assignment_does_not_manufacture_ids_for_missing_track_ids() -> None:
    assigned = add_assignment_temporal_candidate_consensus(
        pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": [0.0, 1.0],
                "source": ["radar", "radar"],
                "track_id": [None, pd.NA],
                "x_m": [0.0, 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
                "confidence": [0.5, 0.5],
            }
        ),
        config=TemporalConsensusConfig(
            max_time_gap_s=1.1,
            max_speed_mps=10.0,
        ),
    ).rows.sort_values("time_s")

    early = assigned.iloc[0]
    late = assigned.iloc[1]
    assert early["candidate_reservoir_temporal_forward_assignment_matched"] == pytest.approx(
        1.0
    )
    assert late["candidate_reservoir_temporal_backward_assignment_matched"] == pytest.approx(
        1.0
    )
    assert early["candidate_temporal_forward_track_id"] == ""
    assert late["candidate_temporal_backward_track_id"] == ""
