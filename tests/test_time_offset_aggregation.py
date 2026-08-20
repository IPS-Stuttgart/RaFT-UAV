import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import aggregate_radar_time_offset_sweep


def _constant_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 5.0],
            "east_m": [0.0, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def _constant_error_radar(error_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0, 1, 2, 3],
            "time_s": [1.0, 2.0, 3.0, 4.0],
            "east_m": [error_m] * 4,
            "north_m": [0.0] * 4,
            "up_m": [0.0] * 4,
        }
    )


def test_radar_offset_sweep_pools_oracle_errors_before_summary_statistics() -> None:
    truth = _constant_truth()
    sweep = aggregate_radar_time_offset_sweep(
        [
            (_constant_error_radar(0.0), truth),
            (_constant_error_radar(10.0), truth),
        ],
        [0.0],
    )

    row = sweep.iloc[0]
    assert row["count"] == pytest.approx(8.0)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["mean_3d_error_m"] == pytest.approx(5.0)
    assert row["rmse_3d_error_m"] == pytest.approx(np.sqrt(50.0))
    assert row["p95_3d_error_m"] == pytest.approx(10.0)
    assert row["std_3d_error_m"] == pytest.approx(5.0)
    assert row["max_3d_error_m"] == pytest.approx(10.0)
    assert row["p95_2d_error_m"] == pytest.approx(10.0)
    assert row["std_2d_error_m"] == pytest.approx(5.0)
