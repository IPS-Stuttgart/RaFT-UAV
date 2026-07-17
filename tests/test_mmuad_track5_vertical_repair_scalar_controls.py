from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 100.0, 2.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    "value",
    [
        np.array([2.0]),
        np.array([[2.0]]),
        np.array([2]),
    ],
)
def test_vertical_repair_rejects_non_scalar_iteration_arrays(value: object) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        repair_track5_vertical_spikes(_submission(), iterations=value)


@pytest.mark.parametrize(
    "parameter",
    [
        "max_vertical_speed_mps",
        "max_neighbor_vertical_speed_mps",
        "max_vertical_residual_m",
        "max_horizontal_speed_mps",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0]),
        np.array([[1.0]]),
    ],
)
def test_vertical_repair_rejects_non_scalar_threshold_arrays(
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{parameter} must be finite and non-negative"):
        repair_track5_vertical_spikes(_submission(), **{parameter: value})


def test_vertical_repair_accepts_zero_dimensional_numeric_scalars() -> None:
    expected, expected_diagnostics = repair_track5_vertical_spikes(
        _submission(),
        iterations=1,
        max_vertical_speed_mps=20.0,
        max_neighbor_vertical_speed_mps=10.0,
        max_vertical_residual_m=15.0,
        max_horizontal_speed_mps=80.0,
    )
    actual, actual_diagnostics = repair_track5_vertical_spikes(
        _submission(),
        iterations=np.array(1.0),
        max_vertical_speed_mps=np.array(20.0),
        max_neighbor_vertical_speed_mps=np.array(10.0),
        max_vertical_residual_m=np.array(15.0),
        max_horizontal_speed_mps=np.array(80.0),
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)
