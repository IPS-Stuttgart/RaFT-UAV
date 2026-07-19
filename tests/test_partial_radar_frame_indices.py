from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.time_offset import aggregate_radar_time_offset_sweep
from raft_uav.evaluation.radar_oracle_diagnostics import (
    nearest_candidate_oracle,
    time_offset_sweep,
)


def _partially_indexed_radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [10, 11, np.nan],
            "track_id": [1, 2, 3],
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [0.0, 100.0, 10.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )


def test_oracle_preserves_valid_indices_when_another_row_is_missing() -> None:
    radar = _partially_indexed_radar()
    truth = _truth()

    selected = nearest_candidate_oracle(radar, truth, max_time_delta_s=0.1)
    direct_sweep = time_offset_sweep(
        radar,
        truth,
        offsets_s=[0.0],
        max_time_delta_s=0.1,
    )
    aggregate_sweep = aggregate_radar_time_offset_sweep(
        [(radar, truth)],
        [0.0],
        max_time_delta_s=0.1,
    )

    assert selected["track_id"].tolist() == [1.0, 2.0, 3.0]
    assert selected["oracle_candidate_rows"].tolist() == [1, 1, 1]
    np.testing.assert_allclose(selected["oracle_error_3d_m"], [0.0, 100.0, 0.0])

    direct = direct_sweep.iloc[0]
    assert direct["count"] == 3.0
    assert direct["coverage"] == pytest.approx(1.0)

    aggregate = aggregate_sweep.iloc[0]
    assert aggregate["count"] == 3.0
    assert aggregate["coverage"] == pytest.approx(1.0)
