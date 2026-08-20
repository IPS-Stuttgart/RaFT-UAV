from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_branch_consensus import attach_candidate_branch_consensus
from raft_uav.mmuad.schema import CandidateFrame


def test_cross_source_support_does_not_cross_physical_flights() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "lidar"],
            "track_id": ["a", "b"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.5, 0.5],
        }
    )

    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.1,
        distance_gate_m=1.0,
    ).rows.set_index("track_id")

    assert augmented["branch_consensus_neighbor_count"].tolist() == [0, 0]
    assert augmented[
        "branch_consensus_nearest_cross_source_distance_m"
    ].isna().all()


def test_pair_advantage_does_not_compare_siblings_across_flights() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["shared"] * 4,
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "time_s": [0.0] * 4,
            "source": ["radar", "lidar", "radar", "camera"],
            "track_id": ["raw", "support-a", "calibrated", "support-b"],
            "candidate_branch": ["raw", "raw", "source_translation", "raw"],
            "mmuad_calibration_origin_row": [7, 8, 7, 9],
            "x_m": [0.0, 0.1, 10.0, 10.4],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "ranker_score": [0.5] * 4,
        }
    )

    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.1,
        distance_gate_m=1.0,
    ).rows.set_index("track_id")

    assert augmented.loc[
        "raw", "branch_consensus_nearest_cross_source_distance_m"
    ] == pytest.approx(0.1)
    assert augmented.loc[
        "calibrated", "branch_consensus_nearest_cross_source_distance_m"
    ] == pytest.approx(0.4)
    assert pd.isna(augmented.loc["raw", "branch_consensus_pair_advantage_m"])
    assert pd.isna(
        augmented.loc["calibrated", "branch_consensus_pair_advantage_m"]
    )


def test_branch_score_normalization_is_flight_local() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["high", "low"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.9, 0.1],
        }
    )

    augmented = attach_candidate_branch_consensus(CandidateFrame(rows)).rows.set_index(
        "track_id"
    )

    assert augmented["branch_consensus_base_score_normalized"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert augmented["branch_consensus_rank_percentile"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
