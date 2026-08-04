from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.radar_oracle_diagnostics import (
    interpolate_truth_positions,
    nearest_candidate_oracle,
    time_offset_sweep,
)


def _truth_with_duplicate_timestamp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0],
            "east_m": [100.0, 0.0, 1.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0],
        }
    )


def test_truth_interpolation_keeps_final_duplicate_sample() -> None:
    positions, valid = interpolate_truth_positions(
        _truth_with_duplicate_timestamp(),
        [0.0, 0.5],
        max_time_delta_s=0.5,
    )

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(positions[:, 0], [0.0, 0.5])
    np.testing.assert_allclose(positions[:, 1:], [[0.0, 5.0], [0.0, 5.0]])


def test_radar_oracle_uses_final_duplicate_truth_sample() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0],
            "track_id": [7],
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [5.0],
        }
    )
    truth = _truth_with_duplicate_timestamp()

    selected = nearest_candidate_oracle(
        radar,
        truth,
        max_time_delta_s=0.1,
    )
    sweep = time_offset_sweep(
        radar,
        truth,
        offsets_s=[0.0],
        max_time_delta_s=0.1,
    )

    assert selected["track_id"].tolist() == [7]
    assert selected.loc[0, "oracle_error_3d_m"] == pytest.approx(0.0)
    assert sweep.loc[0, "count"] == 1.0
    assert sweep.loc[0, "mean_3d_error_m"] == pytest.approx(0.0)
