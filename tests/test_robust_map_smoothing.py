import numpy as np

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.robust_map import RobustMapSmootherConfig
from raft_uav.baselines.smoothing import smooth_tracking_records


def _synthetic_records_and_measurements():
    times = np.arange(8, dtype=float)
    truth = np.column_stack(
        [
            10.0 * times,
            0.5 * times,
            np.zeros_like(times),
            np.full_like(times, 10.0),
            np.full_like(times, 0.5),
            np.zeros_like(times),
        ]
    )
    measurements = []
    records = []
    covariance = np.diag([2.0**2, 2.0**2, 5.0**2])
    record_covariance = np.diag([4.0**2, 4.0**2, 8.0**2, 3.0**2, 3.0**2, 3.0**2])
    for idx, time_s in enumerate(times):
        vector = truth[idx, :3].copy()
        if idx == 3:
            vector[0] += 150.0
            vector[1] -= 120.0
        measurements.append(
            TrackingMeasurement(time_s=time_s, vector=vector, covariance=covariance, source="radar")
        )
        state = truth[idx].copy()
        state[:3] = vector
        records.append(
            {
                "time_s": float(time_s),
                "source": "radar",
                "state": state,
                "covariance": record_covariance.copy(),
                "accepted": True,
                "measurement_dim": 3,
            }
        )
    return truth, records, measurements


def test_robust_map_downweights_single_position_outlier():
    truth, records, measurements = _synthetic_records_and_measurements()
    smoothed = smooth_tracking_records(
        records,
        method="robust-map",
        acceleration_std_mps2=0.5,
        measurements=measurements,
        robust_map_config=RobustMapSmootherConfig(
            loss="cauchy",
            loss_scale=2.0,
            max_iterations=100,
            process_position_floor_m=0.5,
            process_velocity_floor_mps=0.2,
        ),
    )

    before = np.linalg.norm(records[3]["state"][:2] - truth[3, :2])
    after = np.linalg.norm(smoothed[3]["state"][:2] - truth[3, :2])
    assert after < 0.5 * before
    assert smoothed[0]["smoother_method"] == "robust-map"
    assert smoothed[0]["map_matched_measurements"] == len(records)
    assert smoothed[0]["map_final_cost"] <= smoothed[0]["map_initial_cost"]


def test_fixed_lag_map_emits_one_record_per_input():
    _truth, records, measurements = _synthetic_records_and_measurements()
    smoothed = smooth_tracking_records(
        records,
        method="fixed-lag-map",
        acceleration_std_mps2=0.5,
        lag_s=3.0,
        measurements=measurements,
        robust_map_config=RobustMapSmootherConfig(max_iterations=50),
    )

    assert len(smoothed) == len(records)
    assert smoothed[0]["smoother_method"] == "fixed-lag-map"
    assert smoothed[0]["smoother_lag_s"] == 3.0
    assert "map_success" in smoothed[0]


def test_robust_map_falls_back_to_posterior_pseudo_measurements():
    _truth, records, _measurements = _synthetic_records_and_measurements()
    smoothed = smooth_tracking_records(
        records,
        method="robust-map",
        acceleration_std_mps2=0.5,
        robust_map_config=RobustMapSmootherConfig(max_iterations=20),
    )

    assert len(smoothed) == len(records)
    assert smoothed[0]["smoother_method"] == "robust-map"
    assert smoothed[0]["map_matched_measurements"] == len(records)
