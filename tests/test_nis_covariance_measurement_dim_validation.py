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


def test_nis_calibration_excludes_missing_acceptance_flags() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": ["radar", "radar", "radar"],
            "measurement_dim": [3, 3, 3],
            "accepted": [True, np.nan, pd.NA],
            "nis": [6.0, 300.0, 300.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        min_samples=1,
    )

    group = payload["groups"]["radar:3"]
    assert group["count"] == 1
    assert group["applied_scale"] == 2.0


def test_nis_calibration_drops_missing_and_blank_sources() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": ["radar", " radar ", None, np.nan, "", "   "],
            "measurement_dim": [3] * 6,
            "accepted": [True] * 6,
            "nis": [3.0] * 6,
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        min_samples=1,
    )

    assert set(payload["groups"]) == {"radar:3"}
    assert payload["groups"]["radar:3"]["count"] == 2
