import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.nis_covariance import (
    NIS_COVARIANCE_CALIBRATION_SCHEMA,
    covariance_scale_for_source_dim,
    fit_nis_covariance_calibration_from_frame,
    validate_nis_covariance_calibration,
)


def _calibration_payload(*, enabled: object = True, measurement_dim: object = 2):
    return {
        "schema": NIS_COVARIANCE_CALIBRATION_SCHEMA,
        "groups": {
            "rf:2": {
                "source": "rf",
                "measurement_dim": measurement_dim,
                "applied_scale": 4.0,
                "enabled": enabled,
            }
        },
    }


def test_string_false_enabled_flag_cannot_enable_runtime_scaling():
    payload = _calibration_payload(enabled="false")

    with pytest.raises(ValueError, match="enabled"):
        validate_nis_covariance_calibration(payload)
    with pytest.raises(ValueError, match="enabled"):
        covariance_scale_for_source_dim(payload, "rf", 2)


def test_literal_false_enabled_flag_remains_disabled():
    payload = _calibration_payload(enabled=False)

    validate_nis_covariance_calibration(payload)
    assert covariance_scale_for_source_dim(payload, "rf", 2) == 1.0


def test_fractional_serialized_measurement_dimension_is_rejected():
    payload = _calibration_payload(measurement_dim=2.9)

    with pytest.raises(ValueError, match="measurement_dim"):
        validate_nis_covariance_calibration(payload)


def test_fractional_min_samples_is_not_truncated():
    diagnostics = pd.DataFrame(
        {
            "source": ["rf", "rf"],
            "measurement_dim": [2, 2],
            "accepted": [True, True],
            "nis": [2.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="min_samples"):
        fit_nis_covariance_calibration_from_frame(
            diagnostics,
            min_samples=1.9,
        )


@pytest.mark.parametrize("value", [True, np.array([1])])
def test_min_samples_rejects_non_integer_scalar_controls(value):
    diagnostics = pd.DataFrame(
        {
            "source": ["radar"],
            "measurement_dim": [3],
            "accepted": [True],
            "nis": [3.0],
        }
    )

    with pytest.raises(ValueError, match="min_samples"):
        fit_nis_covariance_calibration_from_frame(
            diagnostics,
            min_samples=value,
        )


def test_integral_serialized_min_samples_remains_supported():
    diagnostics = pd.DataFrame(
        {
            "source": ["radar"],
            "measurement_dim": [3],
            "accepted": [True],
            "nis": [3.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        min_samples="1.0",
    )

    assert payload["min_samples"] == 1
    assert payload["groups"]["radar:3"]["enabled"] is True
