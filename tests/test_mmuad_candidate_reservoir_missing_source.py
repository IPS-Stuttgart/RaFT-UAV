from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig, build_candidate_reservoir


def test_build_candidate_reservoir_defaults_missing_source_column() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "track_id": ["low", "high"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "confidence": [0.1, 0.9],
        }
    )

    reservoir = build_candidate_reservoir(
        candidates,
        config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=1,
            per_branch_top_n=0,
            max_candidates_per_frame=2,
        ),
    )

    assert list(reservoir["track_id"]) == ["high"]
    assert set(reservoir["source"]) == {"unknown"}
    assert set(reservoir["candidate_branch"]) == {"unknown"}
    assert "source:unknown" in reservoir.loc[0, "candidate_reservoir_reason"]
