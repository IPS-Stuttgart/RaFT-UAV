from __future__ import annotations

import numpy as np
import pytest

from raft_uav.calibration.nis_covariance import (
    covariance_scale_for_source_dim,
    validate_nis_covariance_calibration,
)


def _payload(enabled: object) -> dict[str, object]:
    return {
        "schema": "raft-uav-nis-covariance-calibration-v1",
        "groups": {
            "radar:3": {
                "source": "radar",
                "measurement_dim": 3,
                "count": 10,
                "method": "mean",
                "statistic": 6.0,
                "target": 3.0,
                "raw_scale": 2.0,
                "applied_scale": 2.0,
                "enabled": enabled,
                "accepted_only": True,
                "quantile": None,
            }
        },
    }


@pytest.mark.parametrize("enabled", ["false", "true", 0, 1, None, [], {}])
def test_nis_calibration_rejects_non_boolean_enabled_flags(enabled: object) -> None:
    payload = _payload(enabled)

    with pytest.raises(ValueError, match="enabled must be a Boolean"):
        validate_nis_covariance_calibration(payload)

    with pytest.raises(ValueError, match="enabled must be a Boolean"):
        covariance_scale_for_source_dim(payload, "radar", 3)


@pytest.mark.parametrize(
    ("enabled", "expected_scale"),
    [
        (False, 1.0),
        (True, 2.0),
        (np.bool_(False), 1.0),
        (np.bool_(True), 2.0),
    ],
)
def test_nis_calibration_preserves_real_boolean_enabled_flags(
    enabled: object,
    expected_scale: float,
) -> None:
    payload = _payload(enabled)

    validate_nis_covariance_calibration(payload)

    assert covariance_scale_for_source_dim(payload, "radar", 3) == expected_scale
