import numpy as np

from raft_uav.baselines.kalman import (
    TrackingMeasurement,
    gate_threshold_from_probability,
    run_async_cv_baseline,
)


def _measurement(time_s: float, x: float, y: float, source: str = "rf") -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.array([x, y]),
        covariance=np.diag([1.0, 1.0]),
        source=source,
    )


def test_gate_threshold_from_probability_matches_chi_square_ordering():
    threshold_2d_95 = gate_threshold_from_probability(0.95, 2)
    threshold_2d_99 = gate_threshold_from_probability(0.99, 2)
    threshold_3d_99 = gate_threshold_from_probability(0.99, 3)

    assert 5.0 < threshold_2d_95 < threshold_2d_99
    assert threshold_3d_99 > threshold_2d_99


def test_large_outlier_is_rejected_when_source_gate_is_tight():
    records = run_async_cv_baseline(
        [
            _measurement(0.0, 0.0, 0.0),
            _measurement(1.0, 1.0, 0.0),
            _measurement(2.0, 10_000.0, 10_000.0),
        ],
        gate_thresholds_by_source={"rf": 5.0},
    )

    assert len(records) == 3
    assert records[-1]["accepted"] is False
    assert records[-1]["nis"] > 5.0
    assert np.linalg.norm(records[-1]["state"][:2]) < 100.0


def test_missing_source_gate_keeps_updates_accepted():
    records = run_async_cv_baseline(
        [
            _measurement(0.0, 0.0, 0.0),
            _measurement(1.0, 10_000.0, 10_000.0),
        ],
        gate_thresholds_by_source={"radar": 1.0},
    )

    assert records[-1]["accepted"] is True
    assert records[-1]["source"] == "rf"
