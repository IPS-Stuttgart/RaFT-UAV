import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.radar_oracle_diagnostics import (
    interpolate_truth_positions,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )


@pytest.mark.parametrize(
    ("query_time_s", "expected_east_m"),
    [
        (-0.5e-9, 0.0),
        (1.0 + 0.5e-9, 10.0),
        (2.0 + 0.5e-9, 20.0),
    ],
)
def test_interpolation_accepts_tolerance_equivalent_samples_from_either_side(
    query_time_s: float,
    expected_east_m: float,
) -> None:
    positions, valid = interpolate_truth_positions(
        _truth(),
        [query_time_s],
        max_time_delta_s=0.0,
    )

    assert valid.tolist() == [True]
    np.testing.assert_allclose(
        positions[0],
        np.array([expected_east_m, 0.0, 0.0]),
    )


def test_interpolation_rejects_query_outside_endpoint_tolerance() -> None:
    positions, valid = interpolate_truth_positions(
        _truth(),
        [2.0 + 2.0e-9],
        max_time_delta_s=0.0,
    )

    assert valid.tolist() == [False]
    assert np.isnan(positions[0]).all()
