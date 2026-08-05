from __future__ import annotations

import numpy as np

from raft_uav.baselines import robust_map
from raft_uav.baselines.kalman import TrackingMeasurement


def _records_and_measurements() -> tuple[
    list[dict[str, object]],
    list[TrackingMeasurement],
]:
    records: list[dict[str, object]] = []
    measurements: list[TrackingMeasurement] = []
    for index in range(3):
        state = np.array([float(index), 0.0, 0.0, 1.0, 0.0, 0.0])
        records.append(
            {
                "time_s": float(index),
                "source": "radar",
                "state": state,
                "covariance": np.eye(6),
                "accepted": True,
                "measurement_dim": 3,
            }
        )
        measurements.append(
            TrackingMeasurement(
                time_s=float(index),
                vector=state[:3],
                covariance=np.eye(3),
                source="radar",
            )
        )
    return records, measurements


def test_batch_robust_map_delegates_to_pyrecest(monkeypatch) -> None:
    records, measurements = _records_and_measurements()
    original = robust_map._pyrecest_robust_map
    calls: list[int] = []

    def wrapped(*args, **kwargs):
        calls.append(len(kwargs["measurements"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(robust_map, "_pyrecest_robust_map", wrapped)
    smoothed = robust_map.robust_map_smooth_records(
        records,
        measurements=measurements,
        acceleration_std_mps2=1.0,
    )

    assert calls == [len(records)]
    assert smoothed[0]["map_solver"] == "pyrecest.robust_linear_gaussian_map"


def test_fixed_lag_robust_map_delegates_to_pyrecest(monkeypatch) -> None:
    records, measurements = _records_and_measurements()
    original = robust_map._pyrecest_fixed_lag_robust_map
    calls: list[float] = []

    def wrapped(*args, **kwargs):
        calls.append(float(kwargs["lag"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(robust_map, "_pyrecest_fixed_lag_robust_map", wrapped)
    smoothed = robust_map.robust_map_smooth_records(
        records,
        measurements=measurements,
        acceleration_std_mps2=1.0,
        lag_s=1.5,
    )

    assert calls == [1.5]
    assert smoothed[0]["smoother_method"] == "fixed-lag-map"
