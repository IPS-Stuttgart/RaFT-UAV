from __future__ import annotations

import pandas as pd

from raft_uav.calibration.bias import bias_training_rows


def test_bias_training_rows_sorts_serialized_truth_times_numerically() -> None:
    truth = pd.DataFrame(
        {
            "time_s": ["10", "2"],
            "east_m": [100.0, 20.0],
            "north_m": [0.0, 5.0],
        }
    )
    measurements = pd.DataFrame(
        {
            "time_s": ["2"],
            "east_m": [23.0],
            "north_m": [9.0],
        }
    )

    rows = bias_training_rows(
        measurements,
        truth,
        source="rf",
        max_time_delta_s=0.0,
    )

    assert len(rows) == 1
    assert rows.loc[0, "bias_truth_time_delta_s"] == 0.0
    assert rows.loc[0, "target_bias_east_m"] == 3.0
    assert rows.loc[0, "target_bias_north_m"] == 4.0
