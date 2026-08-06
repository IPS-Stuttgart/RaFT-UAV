from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_calibration_audit import pair_measurements_to_truth


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )


def test_pair_measurements_skips_nonfinite_measurement_rows() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, np.nan, 1.0, 2.0],
            "east_m": [0.0, 99.0, 1.0, np.nan],
            "north_m": [0.0, 99.0, 0.0, 0.0],
            "up_m": [0.0, 99.0, 0.0, 0.0],
        }
    )

    pairs = pair_measurements_to_truth(
        measurements,
        _truth(),
        max_time_delta_s=0.01,
    )

    np.testing.assert_array_equal(
        pairs.measurement_times_s,
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        pairs.measurement_positions_m,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        pairs.truth_times_s,
        np.array([0.0, 1.0]),
    )


def test_pair_measurements_returns_empty_for_only_unusable_rows() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [np.nan, 2.0],
            "east_m": [0.0, np.nan],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    pairs = pair_measurements_to_truth(measurements, _truth())

    assert pairs.measurement_times_s.shape == (0,)
    assert pairs.measurement_positions_m.shape == (0, 3)
    assert pairs.truth_times_s.shape == (0,)
    assert pairs.truth_positions_m.shape == (0, 3)
