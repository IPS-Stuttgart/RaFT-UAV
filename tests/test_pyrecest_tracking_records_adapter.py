from __future__ import annotations

import numpy as np

from raft_uav.baselines.kalman import TrackingMeasurement, run_async_cv_baseline


def _measurement(time_s: float, vector: list[float], *, source: str = "radar") -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=time_s,
        vector=np.asarray(vector, dtype=float),
        covariance=np.eye(len(vector), dtype=float),
        source=source,
    )


def test_async_cv_records_include_pyrecest_prior_posterior_fields() -> None:
    records = run_async_cv_baseline(
        [
            _measurement(0.0, [0.0, 0.0, 0.0]),
            _measurement(1.0, [1.0, 0.0, 0.0]),
        ],
        acceleration_std_mps2=1.0,
    )

    assert len(records) == 2
    for record in records:
        assert "prior_mean" in record
        assert "prior_cov" in record
        assert "posterior_mean" in record
        assert "posterior_cov" in record
        assert "innovation" in record
        assert "innovation_cov" in record
        assert "measurement" in record
        assert "action" in record
        assert np.allclose(record["state"], record["posterior_mean"])
        assert np.allclose(record["covariance"], record["posterior_cov"])
        assert np.asarray(record["prior_mean"]).shape == (6,)
        assert np.asarray(record["posterior_mean"]).shape == (6,)
        assert np.asarray(record["prior_cov"]).shape == (6, 6)
        assert np.asarray(record["posterior_cov"]).shape == (6, 6)


def test_rejected_measurement_record_keeps_predicted_prior_as_posterior() -> None:
    records = run_async_cv_baseline(
        [
            _measurement(0.0, [0.0, 0.0, 0.0]),
            _measurement(1.0, [100.0, 0.0, 0.0]),
        ],
        gate_thresholds_by_source={"radar": 1.0},
        acceleration_std_mps2=1.0,
    )

    outlier_record = records[-1]
    assert outlier_record["accepted"] is False
    assert outlier_record["update_action"] == "rejected"
    assert outlier_record["action"] == "rejected"
    assert float(outlier_record["nis"]) > 1.0
    assert np.allclose(outlier_record["state"], outlier_record["prior_mean"])
    assert np.allclose(outlier_record["covariance"], outlier_record["prior_cov"])
