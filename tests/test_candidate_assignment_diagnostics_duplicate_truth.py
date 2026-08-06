from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_assignment_diagnostics import (
    CandidateAssignmentDiagnosticsConfig,
    build_candidate_assignment_diagnostics,
)


def test_candidate_assignment_diagnostics_use_final_duplicate_truth_snapshot() -> None:
    assignments = pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar",
                "track_id": "stale",
                "candidate_branch": "raw",
                "x_m": 100.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "mixture_final_weight": 0.9,
                "mixture_dominant": True,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar",
                "track_id": "final",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "mixture_final_weight": 0.1,
                "mixture_dominant": False,
            },
        ]
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": ["0", 1.0, 0.0],
            "x_m": [100.0, 10.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        },
        index=[7, 7, 7],
    )

    frames, _ = build_candidate_assignment_diagnostics(
        assignments,
        truth,
        config=CandidateAssignmentDiagnosticsConfig(
            good_candidate_threshold_m=5.0,
            regret_threshold_m=2.0,
            top_k=1,
        ),
    )

    frame = frames.iloc[0]
    assert frame["truth_x_m"] == 0.0
    assert frame["oracle_track_id"] == "final"
    assert frame["dominant_track_id"] == "stale"
    assert frame["assignment_failure_mode"] == "good_candidate_buried"
