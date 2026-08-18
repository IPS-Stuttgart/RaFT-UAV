from __future__ import annotations

import numpy as np
import pytest

from raft_uav.evaluation.radar_calibration_audit import (
    MeasurementTruthPairs,
    fit_yaw_offset_altitude,
)


def _pairs(measured_xy, truth_xy) -> MeasurementTruthPairs:
    measured_xy_array = np.asarray(measured_xy, dtype=float)
    truth_xy_array = np.asarray(truth_xy, dtype=float)
    assert measured_xy_array.shape == truth_xy_array.shape
    count = measured_xy_array.shape[0]
    times = np.arange(count, dtype=float)
    return MeasurementTruthPairs(
        measurement_times_s=times,
        measurement_positions_m=np.column_stack(
            (measured_xy_array, np.zeros(count, dtype=float))
        ),
        truth_times_s=times.copy(),
        truth_positions_m=np.column_stack(
            (truth_xy_array, np.zeros(count, dtype=float))
        ),
    )


@pytest.mark.parametrize(
    ("measured_xy", "truth_xy"),
    [
        (
            [[1.0, 2.0], [1.0, 2.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ),
        (
            [[0.0, 0.0], [1.0, 0.0]],
            [[3.0, 4.0], [3.0, 4.0]],
        ),
        (
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            [[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        ),
    ],
)
def test_yaw_fit_rejects_unobservable_horizontal_geometry(
    measured_xy,
    truth_xy,
) -> None:
    with pytest.raises(ValueError, match="does not constrain a unique yaw"):
        fit_yaw_offset_altitude(_pairs(measured_xy, truth_xy))


def test_yaw_fit_keeps_valid_two_point_minimum() -> None:
    fit = fit_yaw_offset_altitude(
        _pairs(
            [[0.0, 0.0], [1.0, 0.0]],
            [[10.0, 20.0], [10.0, 21.0]],
        )
    )

    assert fit.yaw_rad == pytest.approx(np.pi / 2.0)
