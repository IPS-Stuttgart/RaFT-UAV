import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import (
    nearest_candidate_oracle,
    time_offset_sweep,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )


def _radar_with_partial_frame_indices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [10.0, np.nan],
            "time_s": [0.5, 1.5],
            "east_m": [0.5, 1.5],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "track_id": [1, 2],
        }
    )


def test_nearest_candidate_oracle_keeps_rows_with_missing_frame_index() -> None:
    selected = nearest_candidate_oracle(
        _radar_with_partial_frame_indices(),
        _truth(),
        max_time_delta_s=1.0,
    )

    assert selected["time_s"].tolist() == [0.5, 1.5]
    assert selected["oracle_error_3d_m"].tolist() == [0.0, 0.0]


def test_time_offset_sweep_counts_unindexed_frames_in_coverage() -> None:
    sweep = time_offset_sweep(
        _radar_with_partial_frame_indices(),
        _truth(),
        [0.0],
        max_time_delta_s=1.0,
    )

    row = sweep.iloc[0]
    assert row["count"] == 2.0
    assert row["coverage"] == 1.0
