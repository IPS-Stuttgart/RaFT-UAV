from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import (
    AsyncConstantVelocityKalmanTracker,
    TrackingMeasurement,
)


def test_tracking_measurement_rejects_complex_vector() -> None:
    vector = np.array([1.0 + 2.0j, 2.0, 3.0])

    with pytest.raises(
        ValueError,
        match="measurement vector must contain only real values",
    ):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_complex_covariance() -> None:
    covariance = np.eye(3, dtype=complex)
    covariance[0, 0] = 1.0 + 2.0j

    with pytest.raises(
        ValueError,
        match="measurement covariance must contain only real values",
    ):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_tracking_measurement_rejects_object_wrapped_complex_vector() -> None:
    vector = np.array([1.0 + 2.0j, 2.0, 3.0], dtype=object)

    with pytest.raises(
        ValueError,
        match="measurement vector must contain only real values",
    ):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_object_wrapped_complex_covariance() -> None:
    covariance = np.eye(3, dtype=object)
    covariance[0, 0] = 1.0 + 2.0j

    with pytest.raises(
        ValueError,
        match="measurement covariance must contain only real values",
    ):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_tracking_measurement_rejects_boolean_vector() -> None:
    with pytest.raises(
        ValueError,
        match="measurement vector must contain only real non-Boolean values",
    ):
        TrackingMeasurement(0.0, [True, 2.0, 3.0], np.eye(3), "radar")


def test_tracking_measurement_rejects_boolean_covariance() -> None:
    with pytest.raises(
        ValueError,
        match="measurement covariance must contain only real non-Boolean values",
    ):
        TrackingMeasurement(0.0, np.zeros(3), np.eye(3, dtype=bool), "radar")


def test_tracking_measurement_rejects_nested_boolean_vector() -> None:
    vector = np.array([np.array(True, dtype=object), 2.0, 3.0], dtype=object)

    with pytest.raises(
        ValueError,
        match="measurement vector must contain only real non-Boolean values",
    ):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_nested_boolean_covariance() -> None:
    covariance = np.eye(3, dtype=object)
    covariance[0, 0] = np.array(True, dtype=object)

    with pytest.raises(
        ValueError,
        match="measurement covariance must contain only real non-Boolean values",
    ):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_cv_tracker_rejects_boolean_initial_position() -> None:
    with pytest.raises(
        ValueError,
        match="initial_position must contain only real non-Boolean values",
    ):
        AsyncConstantVelocityKalmanTracker(
            initial_position=np.array([True, False, True]),
            initial_time_s=0.0,
        )


def test_cv_tracker_rejects_nested_boolean_initial_position() -> None:
    initial_position = np.array(
        [np.array(True, dtype=object), 2.0, 3.0],
        dtype=object,
    )

    with pytest.raises(
        ValueError,
        match="initial_position must contain only real non-Boolean values",
    ):
        AsyncConstantVelocityKalmanTracker(
            initial_position=initial_position,
            initial_time_s=0.0,
        )


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
