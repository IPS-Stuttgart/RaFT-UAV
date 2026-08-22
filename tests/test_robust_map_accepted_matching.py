import numpy as np
import pytest

from raft_uav.baselines import robust_map
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.robust_map import RobustMapSmootherConfig
from raft_uav.baselines.smoothing import smooth_tracking_records


def _record(time_s: float, *, accepted: object) -> dict[str, object]:
    return {
        "time_s": time_s,
        "source": "radar",
        "state": np.zeros(6),
        "covariance": np.eye(6),
        "accepted": accepted,
        "measurement_dim": 3,
    }


def test_accepted_only_matching_skips_rejected_nearest_record():
    records = [
        _record(0.0, accepted=False),
        _record(0.5e-6, accepted=True),
    ]
    measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.zeros(3),
            covariance=np.eye(3),
            source="radar",
        )
    ]

    smoothed = smooth_tracking_records(
        records,
        method="robust-map",
        acceleration_std_mps2=1.0,
        measurements=measurements,
        robust_map_config=RobustMapSmootherConfig(
            accepted_measurements_only=True,
            measurement_time_tolerance_s=1.0e-6,
            max_iterations=5,
        ),
    )

    assert len(smoothed) == len(records)
    assert all(record["map_matched_measurements"] == 1 for record in smoothed)


def test_accepted_only_matching_rejects_truthy_string_flag():
    records = [_record(0.0, accepted="False")]
    measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.zeros(3),
            covariance=np.eye(3),
            source="radar",
        )
    ]

    with pytest.raises(ValueError, match=r"records\[0\]\.accepted"):
        robust_map._matched_measurement_factors(
            records,
            measurements,
            np.array([0.0]),
            time_tolerance_s=1.0e-6,
            accepted_only=True,
        )


def test_accepted_only_pseudo_measurements_reject_truthy_string_flag():
    records = [_record(0.0, accepted="False")]

    with pytest.raises(ValueError, match=r"records\[0\]\.accepted"):
        robust_map._record_pseudo_measurement_factors(
            records,
            np.zeros((1, 6)),
            np.eye(6).reshape(1, 6, 6),
            accepted_only=True,
        )


def test_measurement_matching_maximizes_factor_count_before_time_error():
    records = [
        _record(0.0, accepted=True),
        _record(0.8, accepted=True),
    ]
    measurements = [
        TrackingMeasurement(
            time_s=0.5,
            vector=np.array([1.0, 0.0, 0.0]),
            covariance=np.eye(3),
            source="radar",
        ),
        TrackingMeasurement(
            time_s=1.0,
            vector=np.array([2.0, 0.0, 0.0]),
            covariance=np.eye(3),
            source="radar",
        ),
    ]

    factors = robust_map._matched_measurement_factors(
        records,
        measurements,
        np.array([0.0, 0.8]),
        time_tolerance_s=0.5,
        accepted_only=False,
    )

    assert [(factor.index, float(factor.vector[0])) for factor in factors] == [
        (0, 1.0),
        (1, 2.0),
    ]
