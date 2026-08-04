from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.time_offset import (
    sweep_positions_against_truth,
    truth_positions_at_times,
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


def test_time_offset_uses_final_duplicate_truth_sample() -> None:
    truth = _truth_with_duplicate_timestamp()

    positions, valid = truth_positions_at_times(
        truth,
        np.asarray([0.0, 0.5]),
        max_delta_s=0.5,
    )
    sweep = sweep_positions_against_truth(
        measurement_times_s=np.asarray([0.0]),
        measurement_positions_m=np.asarray([[0.0, 0.0, 5.0]]),
        truth=truth,
        taus_s=[0.0],
        dimensions=3,
        max_truth_time_delta_s=0.5,
    )

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(positions[:, 0], [0.0, 0.5])
    np.testing.assert_allclose(positions[:, 1:], [[0.0, 5.0], [0.0, 5.0]])
    assert sweep.loc[0, "matched_count"] == 1
    assert sweep.loc[0, "mean_error_m"] == pytest.approx(0.0)
