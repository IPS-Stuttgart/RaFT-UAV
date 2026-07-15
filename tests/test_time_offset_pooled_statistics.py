from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import aggregate_radar_time_offset_sweep


def _flight(errors_m: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = np.arange(len(errors_m), dtype=float)
    truth = pd.DataFrame(
        {
            "time_s": times,
            "east_m": np.zeros(len(errors_m)),
            "north_m": np.zeros(len(errors_m)),
            "up_m": np.zeros(len(errors_m)),
        }
    )
    radar = pd.DataFrame(
        {
            "frame_index": np.arange(len(errors_m)),
            "time_s": times,
            "east_m": errors_m,
            "north_m": np.zeros(len(errors_m)),
            "up_m": np.zeros(len(errors_m)),
        }
    )
    return radar, truth


def test_radar_offset_statistics_are_computed_from_pooled_frame_errors() -> None:
    first = _flight([0.0, 0.0, 0.0, 0.0, 100.0])
    second = _flight([0.0])
    pooled = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 0.0])

    sweep = aggregate_radar_time_offset_sweep([first, second], offsets_s=[0.0])

    row = sweep.iloc[0]
    assert row["count"] == 6.0
    assert row["coverage"] == 1.0
    assert row["mean_3d_error_m"] == pytest.approx(np.mean(pooled))
    assert row["std_3d_error_m"] == pytest.approx(np.std(pooled))
    assert row["rmse_3d_error_m"] == pytest.approx(np.sqrt(np.mean(pooled**2)))
    assert row["p95_3d_error_m"] == pytest.approx(np.percentile(pooled, 95.0))
    assert row["max_3d_error_m"] == 100.0
    assert row["std_2d_error_m"] == pytest.approx(np.std(pooled))
    assert row["p95_2d_error_m"] == pytest.approx(np.percentile(pooled, 95.0))
