from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.baselines.radar_association import run_async_cv_baseline_with_radar_association


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("radar_xy_std_m", math.nan),
        ("radar_xy_std_m", math.inf),
        ("radar_z_std_m", -math.inf),
        ("radar_range_std_m", math.nan),
        ("radar_range_std_fraction", math.inf),
        ("radar_crossrange_angle_std_deg", math.nan),
        ("radar_crossrange_min_std_m", math.nan),
        ("radar_crossrange_max_std_m", math.inf),
    ],
)
def test_radar_association_rejects_nonfinite_covariance_parameters(
    parameter: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        run_async_cv_baseline_with_radar_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            association="prediction-nis",
            **{parameter: value},
        )
