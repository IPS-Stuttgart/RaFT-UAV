from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_oracle_gap import build_candidate_oracle_gap


def _truth_rows(times: list[float], x_values: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"] * len(times),
            "time_s": times,
            "x_m": x_values,
            "y_m": [0.0] * len(times),
            "z_m": [0.0] * len(times),
        }
    )


def _candidate_rows(times: list[float], x_values: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"] * len(times),
            "time_s": times,
            "source": ["lidar_360"] * len(times),
            "track_id": [f"track-{index}" for index in range(len(times))],
            "x_m": x_values,
            "y_m": [0.0] * len(times),
            "z_m": [0.0] * len(times),
            "confidence": [0.8] * len(times),
        }
    )


def test_oracle_gap_rejects_boxed_candidate_pseudonumbers() -> None:
    truth = _truth_rows([0.0], [0.0])
    candidates = _candidate_rows(
        [0.0, 0.0, 0.0],
        [
            np.array(0.1 + 4.0j),
            np.array(True, dtype=object),
            2.0,
        ],
    )

    rows = build_candidate_oracle_gap(
        candidates,
        candidates.iloc[[2]].copy(),
        truth,
        max_time_delta_s=0.1,
    )

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["nearest_candidate_track_id"] == "track-2"
    assert int(row["candidate_count_at_nearest_time"]) == 1
    assert float(row["nearest_minus_truth_error_m"]) == 2.0


def test_oracle_gap_rejects_boxed_truth_pseudonumbers() -> None:
    truth = _truth_rows(
        [0.0, 1.0, 2.0],
        [
            0.0,
            np.array(1.0 + 3.0j),
            np.array(True, dtype=object),
        ],
    )
    candidates = _candidate_rows([0.0, 1.0, 2.0], [0.0, 1.0, 1.0])

    rows = build_candidate_oracle_gap(
        candidates,
        candidates,
        truth,
        max_time_delta_s=0.1,
    )

    assert rows["time_s"].tolist() == [0.0]


def test_oracle_gap_preserves_recursively_boxed_real_values() -> None:
    boxed_real = np.empty((), dtype=object)
    boxed_real[()] = np.array(1.5)
    boxed_zero_imaginary = np.empty((), dtype=object)
    boxed_zero_imaginary[()] = np.array(2.0 + 0.0j)
    truth = _truth_rows([0.0], [0.0])
    candidates = _candidate_rows(
        [0.0, 0.0],
        [boxed_real, boxed_zero_imaginary],
    )

    rows = build_candidate_oracle_gap(
        candidates,
        candidates.iloc[[0]].copy(),
        truth,
        max_time_delta_s=0.1,
    )

    row = rows.iloc[0]
    assert int(row["candidate_count_at_nearest_time"]) == 2
    assert float(row["nearest_minus_truth_error_m"]) == 1.5
