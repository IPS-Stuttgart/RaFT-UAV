import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement


def test_tracking_measurement_rejects_complex_vector_before_coercion():
    vector = np.array([1.0 + 2.0j, 2.0 + 0.0j, 3.0 + 0.0j])

    with pytest.raises(ValueError, match="measurement vector.*real"):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_complex_covariance_before_coercion():
    covariance = np.eye(3, dtype=complex)
    covariance[0, 0] = 1.0 + 2.0j

    with pytest.raises(ValueError, match="measurement covariance.*real"):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_tracking_measurement_accepts_real_object_arrays():
    vector = np.array([1.0, 2.0, 3.0], dtype=object)
    covariance = np.array(np.eye(3), dtype=object)

    measurement = TrackingMeasurement(0.0, vector, covariance, "radar")

    np.testing.assert_allclose(measurement.vector, np.asarray(vector, dtype=float))
    np.testing.assert_allclose(
        measurement.covariance,
        np.asarray(covariance, dtype=float),
    )
