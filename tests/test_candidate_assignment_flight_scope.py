from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_assignment_blocks import (
    build_candidate_assignment_block_tables,
)
from raft_uav.mmuad.candidate_assignment_diagnostics import (
    CandidateAssignmentDiagnosticsConfig,
    build_candidate_assignment_diagnostics,
)


def test_candidate_assignment_diagnostics_do_not_pool_physical_flights() -> None:
    assignments = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["1", "2"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["flight-1", "flight-2"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "mixture_final_weight": [1.0, 1.0],
            "mixture_dominant": [True, True],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": [1, 2],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    frames, summary = build_candidate_assignment_diagnostics(
        assignments,
        truth,
        config=CandidateAssignmentDiagnosticsConfig(top_k=1),
    )

    frames = frames.sort_values("flight_id").reset_index(drop=True)
    assert frames["sequence_id"].tolist() == ["shared", "shared"]
    assert frames["flight_id"].tolist() == ["1", "2"]
    assert frames["candidate_count"].tolist() == [1, 1]
    assert frames["truth_x_m"].tolist() == [0.0, 100.0]
    assert frames["oracle_error_3d_m"].tolist() == [0.0, 0.0]
    assert frames["state_error_3d_m"].tolist() == [0.0, 0.0]

    per_flight = summary.loc[
        (summary["sequence_id"] == "shared")
        & (summary["assignment_failure_mode"] == "__all__")
    ].sort_values("flight_id")
    assert per_flight["flight_id"].tolist() == ["1", "2"]
    assert per_flight["frame_count"].tolist() == [1, 1]

    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["assignment_failure_mode"] == "__all__")
    ].iloc[0]
    assert pooled["flight_id"] == "__pooled__"
    assert pooled["frame_count"] == 2


def test_candidate_assignment_blocks_do_not_join_physical_flights() -> None:
    frame_rows = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared", "shared", "shared"],
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "assignment_failure_mode": ["covered", "covered", "covered", "covered"],
        }
    )

    blocks, summary = build_candidate_assignment_block_tables(
        frame_rows,
        max_gap_s=1.0,
    )

    blocks = blocks.sort_values("flight_id").reset_index(drop=True)
    assert blocks["sequence_id"].tolist() == ["shared", "shared"]
    assert blocks["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert blocks["frame_count"].tolist() == [2, 2]
    assert blocks["duration_s"].tolist() == [1.0, 1.0]

    per_flight = summary.loc[
        (summary["sequence_id"] == "shared")
        & (summary["assignment_failure_mode"] == "__all__")
    ].sort_values("flight_id")
    assert per_flight["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert per_flight["frame_count"].tolist() == [2, 2]

    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["assignment_failure_mode"] == "__all__")
    ].iloc[0]
    assert pooled["flight_id"] == "__pooled__"
    assert pooled["frame_count"] == 4
