from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_gap import build_candidate_oracle_gap


def _pooled_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["raw-a", "raw-b"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [1.0, 1.0],
        }
    )


def _pooled_selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["selected-a", "selected-b"],
            "x_m": [100.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [1.0, 1.0],
        }
    )


def _pooled_truth() -> pd.DataFrame:
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


def test_candidate_oracle_gap_isolates_reused_sequence_ids_by_flight() -> None:
    rows = build_candidate_oracle_gap(
        _pooled_candidates(),
        _pooled_selected(),
        _pooled_truth(),
        max_time_delta_s=0.1,
    )

    assert len(rows) == 2
    assert set(rows["flight_id"]) == {"flight-a", "flight-b"}
    assert set(rows["sequence_id"]) == {"shared"}

    by_flight = rows.set_index("flight_id")
    assert float(by_flight.loc["flight-a", "selected_minus_truth_error_m"]) == 100.0
    assert float(by_flight.loc["flight-b", "selected_minus_truth_error_m"]) == 100.0
    assert float(by_flight.loc["flight-a", "nearest_minus_truth_error_m"]) == 0.0
    assert float(by_flight.loc["flight-b", "nearest_minus_truth_error_m"]) == 0.0
    assert float(by_flight.loc["flight-a", "candidate_regret_m"]) == 100.0
    assert float(by_flight.loc["flight-b", "candidate_regret_m"]) == 100.0
    assert int(by_flight.loc["flight-a", "candidate_count_at_nearest_time"]) == 1
    assert int(by_flight.loc["flight-b", "candidate_count_at_nearest_time"]) == 1
    assert by_flight.loc["flight-a", "selected_candidate_track_id"] == "selected-a"
    assert by_flight.loc["flight-b", "selected_candidate_track_id"] == "selected-b"


def test_candidate_oracle_gap_rejects_ambiguous_one_sided_flight_metadata() -> None:
    selected = _pooled_selected().drop(columns="flight_id")
    truth = _pooled_truth().drop(columns="flight_id")

    with pytest.raises(ValueError, match="complete flight_id metadata"):
        build_candidate_oracle_gap(
            _pooled_candidates(),
            selected,
            truth,
            max_time_delta_s=0.1,
        )
