from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.uncertainty import _aligned_residuals, _nearest_time_indices


def test_uncertainty_nearest_time_keeps_final_duplicate_reference() -> None:
    reference_times = np.array([1.0, 0.0, 0.0])
    query_times = np.array([0.0, 0.5, 1.0])

    indices = _nearest_time_indices(reference_times, query_times)

    assert indices.tolist() == [2, 2, 0]


def test_uncertainty_residuals_use_final_duplicate_truth_sample() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [100.0, 0.0, 1.0],
            "north_m": [50.0, 1.0, 1.0],
        }
    )

    aligned = _aligned_residuals(
        frame,
        truth,
        max_time_delta_s=0.1,
    )

    assert aligned["residual_east_m"].tolist() == [0.0]
    assert aligned["residual_north_m"].tolist() == [0.0]


def test_uncertainty_nearest_time_preserves_unique_timestamp_behavior() -> None:
    reference_times = np.array([2.0, 0.0, 1.0])
    query_times = np.array([0.6, 1.8])

    indices = _nearest_time_indices(reference_times, query_times)

    assert indices.tolist() == [2, 0]
