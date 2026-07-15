import numpy as np

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.robust_map import RobustMapSmootherConfig
from raft_uav.baselines.smoothing import smooth_tracking_records


def _record(time_s: float, *, accepted: bool) -> dict[str, object]:
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
