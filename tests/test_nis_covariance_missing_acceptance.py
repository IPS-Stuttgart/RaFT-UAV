import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.nis_covariance import (
    fit_nis_covariance_calibration_from_frame,
)


@pytest.mark.parametrize("missing_acceptance", [np.nan, pd.NA])
def test_accepted_only_calibration_excludes_missing_acceptance(missing_acceptance):
    diagnostics = pd.DataFrame(
        {
            "source": ["radar", "radar"],
            "measurement_dim": [3, 3],
            "accepted": [True, missing_acceptance],
            "nis": [3.0, 300.0],
        }
    )

    payload = fit_nis_covariance_calibration_from_frame(
        diagnostics,
        method="mean",
        min_samples=1,
    )

    group = payload["groups"]["radar:3"]
    assert group["count"] == 1
    assert group["applied_scale"] == 1.0
