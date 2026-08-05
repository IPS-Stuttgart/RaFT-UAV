from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.calibration.nis_covariance import (
    fit_nis_covariance_calibration_from_frame,
)


def test_calibration_drops_missing_sources_before_string_coercion() -> None:
    diagnostics = pd.DataFrame(
        {
            "source": [None, np.nan, "radar"],
            "measurement_dim": [3, 3, 3],
            "accepted": [True, True, True],
            "nis": [30.0, 30.0, 3.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        method="mean",
        min_samples=1,
    )

    assert set(payload["groups"]) == {"radar:3"}
    assert payload["groups"]["radar:3"]["count"] == 1
    assert payload["groups"]["radar:3"]["applied_scale"] == 1.0
