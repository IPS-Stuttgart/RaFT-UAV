from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement


@pytest.mark.parametrize(
    "flag",
    [
        pytest.param("False", id="false-string"),
        pytest.param("True", id="true-string"),
        pytest.param(0, id="integer-zero"),
        pytest.param(1, id="integer-one"),
        pytest.param(None, id="none"),
        pytest.param(np.array(False), id="zero-dimensional-array"),
        pytest.param(np.ma.array(False, mask=True), id="masked-boolean"),
    ],
)
def test_tracking_measurement_rejects_ambiguous_runtime_calibration_flag(flag: object) -> None:
    with pytest.raises(
        ValueError,
        match="_apply_runtime_calibration must be a Boolean scalar",
    ):
        TrackingMeasurement(
            time_s=0.0,
            vector=np.zeros(3),
            covariance=np.eye(3),
            source="radar",
            _apply_runtime_calibration=flag,
        )


@pytest.mark.parametrize("flag", [False, True, np.bool_(False), np.bool_(True)])
def test_tracking_measurement_accepts_boolean_runtime_calibration_flag(flag: object) -> None:
    measurement = TrackingMeasurement(
        time_s=0.0,
        vector=np.zeros(3),
        covariance=np.eye(3),
        source="radar",
        _apply_runtime_calibration=flag,
    )

    assert measurement.time_s == 0.0
    assert measurement.source == "radar"
