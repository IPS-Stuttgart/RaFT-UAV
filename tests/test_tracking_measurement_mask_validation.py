import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement


def test_tracking_measurement_rejects_masked_vector_values():
    vector = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])

    with pytest.raises(ValueError, match="measurement vector.*masked"):
        TrackingMeasurement(0.0, vector, np.eye(3), "radar")


def test_tracking_measurement_rejects_masked_covariance_values():
    covariance = np.ma.array(np.eye(3), mask=False)
    covariance.mask[1, 1] = True

    with pytest.raises(ValueError, match="measurement covariance.*masked"):
        TrackingMeasurement(0.0, np.zeros(3), covariance, "radar")


def test_tracking_measurement_accepts_explicitly_unmasked_arrays():
    vector = np.ma.array([1.0, 2.0, 3.0], mask=False)
    covariance = np.ma.array(np.eye(3), mask=False)

    measurement = TrackingMeasurement(0.0, vector, covariance, "radar")

    np.testing.assert_allclose(measurement.vector, np.asarray(vector))
    np.testing.assert_allclose(measurement.covariance, np.asarray(covariance))
