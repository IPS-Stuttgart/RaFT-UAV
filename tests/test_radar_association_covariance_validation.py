from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import radar_association
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
        ("candidate_catprob_threshold", math.nan),
        ("paper_compatible_catprob_threshold", math.inf),
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


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("candidate_catprob_threshold", -0.01),
        ("candidate_catprob_threshold", 1.01),
        ("paper_compatible_catprob_threshold", -0.01),
        ("paper_compatible_catprob_threshold", 1.01),
    ],
)
def test_radar_association_rejects_out_of_range_catprob_thresholds(
    parameter: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{parameter} must be in \[0, 1\]"):
        run_async_cv_baseline_with_radar_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            association="prediction-nis",
            **{parameter: value},
        )


def test_radar_association_normalizes_valid_scalar_controls() -> None:
    bound = radar_association._RUN_SIGNATURE.bind(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="prediction-nis",
        radar_range_std_m="12.5",
        radar_range_std_fraction=np.array(0.005),
        geometry_velocity_weight=np.ma.array(0.25, mask=False),
        stable_segment_range_gate_m="800",
        candidate_catprob_threshold="0.5",
    )
    bound.apply_defaults()

    radar_association._validate_radar_association_parameters(bound.arguments)

    assert bound.arguments["radar_range_std_m"] == 12.5
    assert isinstance(bound.arguments["radar_range_std_m"], float)
    assert bound.arguments["radar_range_std_fraction"] == 0.005
    assert isinstance(bound.arguments["radar_range_std_fraction"], float)
    assert bound.arguments["geometry_velocity_weight"] == 0.25
    assert isinstance(bound.arguments["geometry_velocity_weight"], float)
    assert bound.arguments["stable_segment_range_gate_m"] == 800.0
    assert isinstance(bound.arguments["stable_segment_range_gate_m"], float)
    assert bound.arguments["candidate_catprob_threshold"] == 0.5
    assert isinstance(bound.arguments["candidate_catprob_threshold"], float)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("radar_range_std_m", True),
        ("geometry_velocity_weight", False),
        ("candidate_catprob_threshold", np.bool_(True)),
        ("stable_segment_range_gate_m", np.array([800.0])),
        ("pda_nis_temperature", np.array(1.0 + 1.0j, dtype=object)),
        ("rf_anchor_weight", np.ma.masked),
    ],
)
def test_radar_association_rejects_lossy_scalar_controls(
    parameter: str,
    value: object,
) -> None:
    bound = radar_association._RUN_SIGNATURE.bind(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="prediction-nis",
        **{parameter: value},
    )
    bound.apply_defaults()

    with pytest.raises(ValueError, match=f"{parameter} must be finite"):
        radar_association._validate_radar_association_parameters(bound.arguments)
