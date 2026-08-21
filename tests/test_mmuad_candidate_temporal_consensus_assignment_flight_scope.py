from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import TemporalConsensusConfig
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
)
from raft_uav.mmuad.schema import CandidateFrame


def _pooled_crossing_flights() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["shared", "shared", "shared", "shared"],
                "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
                "time_s": [0.0, 1.0, 0.0, 1.0],
                "source": ["radar", "radar", "radar", "radar"],
                "track_id": ["a-0", "a-1", "b-0", "b-1"],
                "x_m": [0.0, 100.0, 100.0, 0.0],
                "y_m": [0.0, 0.0, 0.0, 0.0],
                "z_m": [5.0, 5.0, 5.0, 5.0],
                "confidence": [0.2, 0.2, 0.8, 0.8],
                "ranker_score": [0.2, 0.2, 0.8, 0.8],
            }
        )
    )


@pytest.mark.parametrize("assignment_mode", ["nearest", "one-to-one"])
def test_assignment_temporal_consensus_does_not_cross_flights(
    assignment_mode: str,
) -> None:
    result = add_assignment_temporal_candidate_consensus(
        _pooled_crossing_flights(),
        config=TemporalConsensusConfig(
            max_time_gap_s=1.1,
            max_speed_mps=10.0,
        ),
        assignment_mode=assignment_mode,
    ).rows

    assert result["time_s"].is_monotonic_increasing
    result = result.sort_values(["flight_id", "time_s"]).reset_index(drop=True)
    assert result["sequence_id"].eq("shared").all()
    assert result["flight_id"].tolist() == [
        "flight-a",
        "flight-a",
        "flight-b",
        "flight-b",
    ]
    assert result["candidate_reservoir_temporal_base_score"].tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
    for direction in ("backward", "forward"):
        assert result[
            f"candidate_reservoir_temporal_{direction}_distance_m"
        ].isna().all()
        assert result[
            f"candidate_reservoir_temporal_{direction}_support_count"
        ].eq(0.0).all()
        assert result[
            f"candidate_reservoir_temporal_{direction}_assignment_matched"
        ].eq(0.0).all()
