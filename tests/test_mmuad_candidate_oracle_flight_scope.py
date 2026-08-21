from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)
from raft_uav.mmuad.candidate_oracle_blocks import build_candidate_oracle_block_tables


def _crossed_flight_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["a", "b"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [0.1, 0.9],
        }
    )


def _crossed_flight_truth() -> pd.DataFrame:
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


def test_candidate_oracle_attribution_is_scoped_by_physical_flight() -> None:
    frame_rows, pooled, _, _ = build_candidate_oracle_attribution_tables(
        _crossed_flight_candidates(),
        _crossed_flight_truth(),
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
    )

    assert frame_rows[["sequence_id", "flight_id"]].to_dict("records") == [
        {"sequence_id": "shared", "flight_id": "flight-a"},
        {"sequence_id": "shared", "flight_id": "flight-b"},
    ]
    assert frame_rows["candidate_count"].tolist() == [1, 1]
    assert frame_rows["oracle_all_3d_m"].tolist() == [0.0, 0.0]
    assert pooled.loc[0, "frame_count"] == 2
    assert pooled.loc[0, "oracle_all_3d_m_mse"] == 0.0


def test_candidate_oracle_attribution_rejects_ambiguous_one_sided_flights() -> None:
    candidates = _crossed_flight_candidates().drop(columns=["flight_id"])

    with pytest.raises(ValueError, match="one-sided flight_id metadata"):
        build_candidate_oracle_attribution_tables(
            candidates,
            _crossed_flight_truth(),
            top_k_values=(1,),
        )


def test_candidate_oracle_attribution_accepts_unambiguous_one_sided_flight() -> None:
    candidates = _crossed_flight_candidates().iloc[[0]].drop(columns=["flight_id"])
    truth = _crossed_flight_truth().iloc[[0]]

    frame_rows, _, _, _ = build_candidate_oracle_attribution_tables(
        candidates,
        truth,
        top_k_values=(1,),
    )

    assert frame_rows["flight_id"].tolist() == ["flight-a"]
    assert frame_rows["oracle_all_3d_m"].tolist() == [0.0]


def test_candidate_oracle_blocks_do_not_merge_cotimestamped_flights() -> None:
    frame_rows = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared", "shared", "shared"],
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "candidate_count": [1, 1, 1, 1],
            "oracle_all_3d_m": [0.0, 0.0, 0.0, 0.0],
            "oracle_all_rank": [1, 1, 1, 1],
            "oracle_in_top1": [True, True, True, True],
        }
    )

    blocks, summary = build_candidate_oracle_block_tables(
        frame_rows,
        top_k=1,
        max_gap_s=1.0,
    )

    assert blocks["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert blocks["block_id"].tolist() == [0, 0]
    assert blocks["frame_count"].tolist() == [2, 2]
    per_flight = summary.loc[summary["sequence_id"] == "shared"]
    assert per_flight[["flight_id", "frame_count"]].to_dict("records") == [
        {"flight_id": "flight-a", "frame_count": 2},
        {"flight_id": "flight-b", "frame_count": 2},
    ]
