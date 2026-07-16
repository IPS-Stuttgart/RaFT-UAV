from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.nis_covariance import (
    fit_nis_covariance_calibration_from_frame,
)


def _diagnostics(measurement_dim: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["radar"],
            "measurement_dim": [measurement_dim],
            "accepted": [True],
            "nis": [3.0],
        }
    )


@pytest.mark.parametrize(
    "measurement_dim",
    [2.9, "3.5", True, 2 + 0j, np.inf, -np.inf],
)
def test_nis_calibration_rejects_noninteger_measurement_dimensions(
    measurement_dim: object,
) -> None:
    with pytest.raises(ValueError, match="measurement_dim values must be"):
        fit_nis_covariance_calibration_from_frame(
            _diagnostics(measurement_dim),
            min_samples=1,
        )


def test_nis_calibration_accepts_exact_integer_string_dimensions() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": ["rf", "radar"],
            "measurement_dim": ["2", "3.0"],
            "accepted": [True, True],
            "nis": [2.0, 3.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        min_samples=1,
    )

    assert set(payload["groups"]) == {"rf:2", "radar:3"}
