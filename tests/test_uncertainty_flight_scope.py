from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.uncertainty import _aligned_residuals


def test_uncertainty_residuals_do_not_cross_physical_flights() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "east_m": [1.0, 101.0],
            "north_m": [1.0, 101.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 100.0],
        }
    )

    aligned = _aligned_residuals(frame, truth, max_time_delta_s=0.1)

    assert aligned["flight_id"].tolist() == ["flight_a", "flight_b"]
    assert aligned["residual_east_m"].tolist() == [1.0, 1.0]
    assert aligned["residual_north_m"].tolist() == [1.0, 1.0]


def test_uncertainty_residuals_scope_by_flight_id_without_sequence_id() -> None:
    frame = pd.DataFrame(
        {
            "flight_id": ["flight_b", "flight_a"],
            "time_s": [0.0, 0.0],
            "east_m": [101.0, 1.0],
            "north_m": [101.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 100.0],
        }
    )

    aligned = _aligned_residuals(frame, truth, max_time_delta_s=0.1)

    assert aligned["flight_id"].tolist() == ["flight_b", "flight_a"]
    assert aligned["residual_east_m"].tolist() == [1.0, 1.0]
    assert aligned["residual_north_m"].tolist() == [1.0, 1.0]


def test_uncertainty_residuals_reject_ambiguous_one_sided_flight_ids() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "east_m": [1.0, 101.0],
            "north_m": [1.0, 101.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 100.0],
        }
    )

    with pytest.raises(ValueError, match="one-sided flight_id metadata"):
        _aligned_residuals(frame, truth, max_time_delta_s=0.1)


def test_uncertainty_residuals_allow_functional_one_sided_flight_ids() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b"],
            "flight_id": ["flight_a", "flight_b"],
            "time_s": [0.0, 0.0],
            "east_m": [1.0, 101.0],
            "north_m": [1.0, 101.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 100.0],
        }
    )

    aligned = _aligned_residuals(frame, truth, max_time_delta_s=0.1)

    assert aligned["residual_east_m"].tolist() == [1.0, 1.0]
    assert aligned["residual_north_m"].tolist() == [1.0, 1.0]
