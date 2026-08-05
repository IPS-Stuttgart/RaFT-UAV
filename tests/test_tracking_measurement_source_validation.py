from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.kalman import TrackingMeasurement

_VECTOR = np.array([1.0, 2.0, 3.0])
_COVARIANCE = np.eye(3)


def _measurement(source: object) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=1.0,
        vector=_VECTOR,
        covariance=_COVARIANCE,
        source=source,
        _apply_runtime_calibration=False,
    )


@pytest.mark.parametrize(
    "source",
    [
        None,
        np.nan,
        pd.NA,
        np.ma.masked,
        "",
        "   ",
        np.array(["radar"]),
        1,
        b"radar",
    ],
)
def test_tracking_measurement_rejects_invalid_sources(source: object) -> None:
    with pytest.raises(
        ValueError,
        match="measurement source must be a non-empty, non-missing string scalar",
    ):
        _measurement(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("radar", "radar"),
        ("  radar  ", "radar"),
        (np.str_("rf"), "rf"),
        (np.array("custom"), "custom"),
    ],
)
def test_tracking_measurement_normalizes_valid_source_labels(
    source: object,
    expected: str,
) -> None:
    measurement = _measurement(source)

    assert measurement.source == expected
