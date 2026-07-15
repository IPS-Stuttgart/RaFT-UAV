from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.baselines.radar_association import (
    run_async_cv_baseline_with_radar_association,
)


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
        ("track_switch_nis_ratio", math.nan),
        ("geometry_velocity_std_mps", math.inf),
        ("geometry_velocity_weight", math.nan),
        ("rf_anchor_time_gate_s", -math.inf),
        ("pda_nis_temperature", math.nan),
        ("track_bank_clutter_intensity", math.inf),
        ("stable_segment_max_transition_speed_mps", math.nan),
        ("stable_segment_range_gate_m", math.inf),
    ],
)
def test_radar_association_rejects_nonfinite_numeric_parameters(
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
