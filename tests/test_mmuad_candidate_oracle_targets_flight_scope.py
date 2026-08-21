from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets


def _two_flight_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "track_id": ["a", "b"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.5, 0.5],
        }
    )


def _two_flight_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def test_candidate_oracle_targets_are_scoped_by_physical_flight() -> None:
    target_rows, frame_summary, summary = build_candidate_oracle_targets(
        _two_flight_candidates(),
        _two_flight_truth(),
    )

    target_rows = target_rows.sort_values("flight_id", kind="mergesort").reset_index(drop=True)
    frame_summary = frame_summary.sort_values("flight_id", kind="mergesort").reset_index(drop=True)

    assert target_rows["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert target_rows["candidate_error_3d_m"].tolist() == pytest.approx([0.0, 0.0])
    assert target_rows["candidate_is_oracle"].tolist() == [True, True]
    assert frame_summary["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert frame_summary["oracle_error_3d_m"].tolist() == pytest.approx([0.0, 0.0])
    assert summary["frame_count"] == 2
    assert {(row["sequence_id"], row["flight_id"]) for row in summary["by_sequence"]} == {
        ("shared", "flight-a"),
        ("shared", "flight-b"),
    }


def test_candidate_oracle_targets_reject_ambiguous_one_sided_flight_metadata() -> None:
    truth = _two_flight_truth().drop(columns=["flight_id"])

    with pytest.raises(ValueError, match="one-sided flight_id metadata is ambiguous"):
        build_candidate_oracle_targets(_two_flight_candidates(), truth)


def test_candidate_oracle_targets_preserve_unambiguous_one_sided_flight_metadata() -> None:
    candidates = _two_flight_candidates().iloc[[0]].reset_index(drop=True)
    truth = _two_flight_truth().iloc[[0]].drop(columns=["flight_id"]).reset_index(drop=True)

    target_rows, frame_summary, summary = build_candidate_oracle_targets(candidates, truth)

    assert target_rows["flight_id"].tolist() == ["flight-a"]
    assert frame_summary["flight_id"].tolist() == ["flight-a"]
    assert summary["by_sequence"][0]["flight_id"] == "flight-a"


def test_candidate_oracle_targets_reject_partial_flight_metadata() -> None:
    candidates = _two_flight_candidates().copy()
    candidates.loc[1, "flight_id"] = None

    with pytest.raises(ValueError, match="flight_id metadata must each be complete or absent"):
        build_candidate_oracle_targets(candidates, _two_flight_truth())
