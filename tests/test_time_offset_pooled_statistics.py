import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import aggregate_radar_time_offset_sweep


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


def _radar(error_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0],
            "time_s": [0.0],
            "east_m": [error_m],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


def test_radar_offset_aggregate_includes_between_flight_variation():
    sweep = aggregate_radar_time_offset_sweep(
        [(_radar(0.0), _truth()), (_radar(10.0), _truth())],
        offsets_s=[0.0],
    )

    row = sweep.iloc[0]
    assert row["count"] == 2.0
    assert row["mean_3d_error_m"] == pytest.approx(5.0)
    assert row["rmse_3d_error_m"] == pytest.approx(np.sqrt(50.0))
    assert row["std_3d_error_m"] == pytest.approx(5.0)
    assert row["std_2d_error_m"] == pytest.approx(5.0)
