from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_branch_consensus import (
    attach_candidate_branch_consensus,
)
from raft_uav.mmuad.schema import CandidateFrame


def _candidate(
    *,
    flight_id: str,
    source: str,
    track_id: str,
    time_s: float,
    x_m: float,
    ranker_score: float,
    branch: str = "raw",
    origin: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sequence_id": "shared",
        "flight_id": flight_id,
        "time_s": time_s,
        "source": source,
        "track_id": track_id,
        "candidate_branch": branch,
        "x_m": x_m,
        "y_m": 0.0,
        "z_m": 0.0,
        "ranker_score": ranker_score,
    }
    if origin is not None:
        row["mmuad_calibration_origin_row"] = origin
    return row


def test_branch_consensus_does_not_use_cross_flight_sensor_support() -> None:
    rows = pd.DataFrame(
        [
            _candidate(
                flight_id="flight-a",
                source="rf",
                track_id="a-rf",
                time_s=0.0,
                x_m=0.0,
                ranker_score=0.5,
            ),
            _candidate(
                flight_id="flight-b",
                source="radar",
                track_id="b-radar",
                time_s=0.0,
                x_m=0.1,
                ranker_score=0.5,
            ),
        ]
    )

    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.1,
        distance_gate_m=5.0,
    ).rows.set_index("flight_id")

    for flight_id in ("flight-a", "flight-b"):
        assert augmented.loc[flight_id, "branch_consensus_neighbor_count"] == 0
        assert augmented.loc[flight_id, "branch_consensus_unique_source_count"] == 0
        assert pd.isna(
            augmented.loc[
                flight_id,
                "branch_consensus_nearest_cross_source_distance_m",
            ]
        )
        assert augmented.loc[flight_id, "branch_consensus_score"] == pytest.approx(0.0)


def test_branch_consensus_normalizes_scores_inside_each_physical_flight() -> None:
    rows = pd.DataFrame(
        [
            _candidate(
                flight_id="flight-a",
                source="rf",
                track_id="a-low",
                time_s=0.0,
                x_m=0.0,
                ranker_score=0.0,
            ),
            _candidate(
                flight_id="flight-a",
                source="rf",
                track_id="a-high",
                time_s=1.0,
                x_m=1.0,
                ranker_score=10.0,
            ),
            _candidate(
                flight_id="flight-b",
                source="rf",
                track_id="b-low",
                time_s=0.0,
                x_m=100.0,
                ranker_score=100.0,
            ),
            _candidate(
                flight_id="flight-b",
                source="rf",
                track_id="b-high",
                time_s=1.0,
                x_m=101.0,
                ranker_score=110.0,
            ),
        ]
    )

    augmented = attach_candidate_branch_consensus(CandidateFrame(rows)).rows.set_index(
        "track_id"
    )

    assert augmented.loc[
        ["a-low", "a-high", "b-low", "b-high"],
        "branch_consensus_base_score_normalized",
    ].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert augmented.loc[
        ["a-low", "a-high", "b-low", "b-high"],
        "branch_consensus_rank_percentile",
    ].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_branch_pair_advantage_does_not_compare_reused_origins_across_flights() -> None:
    rows = pd.DataFrame(
        [
            _candidate(
                flight_id="flight-a",
                source="rf",
                track_id="a-raw",
                time_s=0.0,
                x_m=0.0,
                ranker_score=0.5,
                branch="raw",
                origin=7,
            ),
            _candidate(
                flight_id="flight-a",
                source="lidar",
                track_id="a-lidar",
                time_s=0.0,
                x_m=0.1,
                ranker_score=0.5,
                origin=12,
            ),
            _candidate(
                flight_id="flight-b",
                source="rf",
                track_id="b-calibrated",
                time_s=0.0,
                x_m=100.0,
                ranker_score=0.5,
                branch="source_translation",
                origin=7,
            ),
        ]
    )

    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.1,
        distance_gate_m=5.0,
    ).rows.set_index("track_id")

    assert augmented.loc[
        ["a-raw", "b-calibrated"],
        "branch_consensus_pair_advantage_m",
    ].isna().all()
