from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
)


def _pooled_candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared", "shared", "shared"],
            "flight_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "time_s": [0.0, 0.0, 0.0, 0.0],
            "source": ["radar", "radar", "radar", "radar"],
            "track_id": ["a-high", "a-low", "b-high", "b-low"],
            "candidate_branch": ["raw", "raw", "raw", "raw"],
            "x_m": [0.0, 1.0, 100.0, 101.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "confidence": [0.9, 0.1, 0.8, 0.2],
        }
    )


def _crossed_reservoir_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["candidate-a", "candidate-b"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [1.0, 1.0],
        }
    )


def _flight_truth_rows() -> pd.DataFrame:
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


def test_candidate_reservoir_caps_each_physical_flight_independently() -> None:
    rows = _pooled_candidate_rows()

    reservoir = build_candidate_reservoir(
        rows,
        config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=1,
        ),
    )

    assert set(reservoir["track_id"]) == {"a-high", "b-high"}
    ranks = reservoir.groupby("flight_id")["candidate_reservoir_rank"].apply(list).to_dict()
    assert ranks == {"flight-a": [1.0], "flight-b": [1.0]}

    summary = build_reservoir_summary(rows, reservoir)
    assert summary["input_frame_count"] == 2
    assert summary["reservoir_frame_count"] == 2


def test_candidate_reservoir_oracle_stays_inside_physical_flight() -> None:
    frame_rows, pooled, by_sequence = build_oracle_recall_tables(
        _crossed_reservoir_rows(),
        _flight_truth_rows(),
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
    )

    assert len(frame_rows) == 2
    assert set(frame_rows["flight_id"]) == {"flight-a", "flight-b"}
    assert sorted(frame_rows["oracle_all_3d_m"]) == [100.0, 100.0]
    assert int(pooled.loc[0, "frame_count"]) == 2
    assert float(pooled.loc[0, "oracle_all_3d_m_mse"]) == 10000.0
    assert set(by_sequence["flight_id"]) == {"flight-a", "flight-b"}


def test_candidate_reservoir_oracle_rejects_ambiguous_one_sided_flights() -> None:
    truth = _flight_truth_rows().drop(columns=["flight_id"])

    with pytest.raises(ValueError, match="one-sided flight_id metadata is ambiguous"):
        build_oracle_recall_tables(
            _crossed_reservoir_rows(),
            truth,
            top_k_values=(1,),
            max_truth_time_delta_s=0.1,
        )


def test_candidate_reservoir_oracle_keeps_unambiguous_one_sided_flight() -> None:
    reservoir = _crossed_reservoir_rows().iloc[[0]].copy()
    truth = _flight_truth_rows().iloc[[0]].drop(columns=["flight_id"]).copy()

    frame_rows, _, by_sequence = build_oracle_recall_tables(
        reservoir,
        truth,
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
    )

    assert frame_rows["flight_id"].tolist() == ["flight-a"]
    assert by_sequence["flight_id"].tolist() == ["flight-a"]
