from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables


def test_oracle_recall_selects_exact_extreme_timestamp_without_overflow() -> None:
    magnitude = 1.0e308
    reservoir = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [magnitude],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "candidate_reservoir_score": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [-magnitude, magnitude],
            "x_m": [10.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir,
            truth,
            top_k_values=(1,),
            max_truth_time_delta_s=0.0,
        )

    assert len(frame_rows) == 1
    assert frame_rows.loc[0, "truth_time_delta_s"] == 0.0
    assert frame_rows.loc[0, "oracle_all_3d_m"] == 0.0
    assert pooled.loc[0, "oracle_all_3d_m_mse"] == 0.0
    assert by_sequence.loc[0, "sequence_id"] == "seq"


def test_oracle_recall_keeps_representable_large_distance_finite() -> None:
    reservoir = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [6.0e307],
            "y_m": [8.0e307],
            "z_m": [0.0],
            "candidate_reservoir_score": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        frame_rows, pooled, _ = build_oracle_recall_tables(
            reservoir,
            truth,
            top_k_values=(1,),
            max_truth_time_delta_s=0.0,
        )

    expected = 1.0e308
    assert frame_rows.loc[0, "oracle_all_3d_m"] == pytest.approx(expected)
    assert frame_rows.loc[0, "oracle_top1_3d_m"] == pytest.approx(expected)
    assert np.isinf(pooled.loc[0, "oracle_all_3d_m_mse"])
    assert np.isinf(pooled.loc[0, "oracle_all_3d_m_rmse"])
