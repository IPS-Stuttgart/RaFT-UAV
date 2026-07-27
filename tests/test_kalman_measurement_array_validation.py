from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement


def test_tracking_measurement_rejects_masked_vector() -> None:
    vector = np.ma.array(
        [1_000.0, 2.0, 3.0],
        mask=[True, False, False],
    )

    with pytest.raises(
        ValueError,
        match="measurement vector must contain only unmasked real values",
    ):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_masked_covariance() -> None:
    covariance = np.ma.array(
        np.eye(3),
        mask=[
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
    )

    with pytest.raises(
        ValueError,
        match="measurement covariance must contain only unmasked real values",
    ):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_tracking_measurement_rejects_complex_vector() -> None:
    vector = np.array([1.0 + 2.0j, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="measurement vector must contain only unmasked real values",
    ):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_complex_covariance() -> None:
    covariance = np.eye(3, dtype=complex)
    covariance[0, 0] = 1.0 + 2.0j

    with pytest.raises(
        ValueError,
        match="measurement covariance must contain only unmasked real values",
    ):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_tracking_measurement_preserves_valid_real_arrays() -> None:
    measurement = TrackingMeasurement(
        np.float64(1.25),
        np.array([1.0, 2.0, 3.0], dtype=object),
        np.eye(3, dtype=object),
        "radar",
    )

    assert measurement.time_s == 1.25
    assert measurement.vector.dtype == float
    assert measurement.covariance.dtype == float
