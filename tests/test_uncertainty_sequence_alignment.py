from __future__ import annotations

import pandas as pd

from raft_uav.uncertainty import _aligned_residuals


def test_uncertainty_residuals_do_not_cross_sequence_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["seq_b", "seq_a"],
            "time_s": [0.0, 0.0],
            "east_m": [101.0, 2.0],
            "north_m": [99.0, 3.0],
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

    assert aligned["sequence_id"].tolist() == ["seq_b", "seq_a"]
    assert aligned["residual_east_m"].tolist() == [1.0, 2.0]
    assert aligned["residual_north_m"].tolist() == [-1.0, 3.0]


def test_uncertainty_residuals_normalize_and_require_valid_sequence_ids() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["nan", " seq_a "],
            "time_s": [0.0, 0.0],
            "east_m": [1001.0, 11.0],
            "north_m": [999.0, 19.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": [None, "seq_a"],
            "time_s": [0.0, 0.0],
            "east_m": [1000.0, 10.0],
            "north_m": [1000.0, 20.0],
        }
    )

    aligned = _aligned_residuals(frame, truth, max_time_delta_s=0.1)

    assert aligned["sequence_id"].tolist() == [" seq_a "]
    assert aligned["residual_east_m"].tolist() == [1.0]
    assert aligned["residual_north_m"].tolist() == [-1.0]
