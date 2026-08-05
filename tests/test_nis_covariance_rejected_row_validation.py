from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.nis_covariance import (
    fit_nis_covariance_calibration_from_frame,
)


def _diagnostics_with_malformed_rejected_dimensions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["radar"] * 4,
            "measurement_dim": [3, 3.5, np.inf, True],
            "accepted": [True, False, np.nan, pd.NA],
            "nis": [3.0, 300.0, 300.0, 300.0],
        }
    )


def test_accepted_only_calibration_ignores_malformed_rejected_dimensions() -> None:
    payload = fit_nis_covariance_calibration_from_frame(
        _diagnostics_with_malformed_rejected_dimensions(),
        min_samples=1,
    )

    assert set(payload["groups"]) == {"radar:3"}
    group = payload["groups"]["radar:3"]
    assert group["count"] == 1
    assert group["applied_scale"] == 1.0


def test_all_update_calibration_still_rejects_malformed_dimensions() -> None:
    with pytest.raises(ValueError, match="measurement_dim"):
        fit_nis_covariance_calibration_from_frame(
            _diagnostics_with_malformed_rejected_dimensions(),
            min_samples=1,
            accepted_only=False,
        )
