from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.nis_covariance import (
    fit_nis_covariance_calibration_from_frame,
)


def test_missing_acceptance_is_not_treated_as_accepted() -> None:
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


def test_invalid_sources_and_dimensions_are_discarded_before_grouping() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": [
                "radar",
                " radar ",
                None,
                np.nan,
                "",
                "   ",
                "radar",
                "radar",
                "radar",
            ],
            "measurement_dim": [3, 3, 3, 3, 3, 3, np.inf, 3.5, -1],
            "accepted": [True] * 9,
            "nis": [6.0] * 9,
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        min_samples=1,
    )

    assert set(payload["groups"]) == {"radar:3"}
    group = payload["groups"]["radar:3"]
    assert group["count"] == 2
    assert group["applied_scale"] == 2.0
