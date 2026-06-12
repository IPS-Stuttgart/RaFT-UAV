from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.time_offset import aggregate_radar_time_offset_sweep


def test_radar_time_offset_coverage_counts_zero_match_frames() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [10.0, 10.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [100.0, 101.0, 102.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [10.0, 10.0, 10.0],
        }
    )

    sweep = aggregate_radar_time_offset_sweep(
        [(radar, truth)],
        [0.0],
        max_time_delta_s=0.5,
    )

    row = sweep.iloc[0]
    assert row["count"] == 0.0
    assert row["coverage"] == 0.0
    assert np.isnan(row["mean_3d_error_m"])


def test_radar_time_offset_coverage_uses_frame_index_denominator() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0, 1, 1],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "east_m": [0.0, 50.0, 1.0, 51.0],
            "north_m": [0.0, 50.0, 0.0, 50.0],
            "up_m": [10.0, 10.0, 10.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [10.0],
        }
    )

    sweep = aggregate_radar_time_offset_sweep(
        [(radar, truth)],
        [0.0],
        max_time_delta_s=0.1,
    )

    row = sweep.iloc[0]
    assert row["count"] == 1.0
    assert row["coverage"] == 0.5
