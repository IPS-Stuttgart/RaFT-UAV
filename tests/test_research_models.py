from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import smooth_position_trajectory
from raft_uav.research.measurement_models import enu_covariance_from_range_az_el
from raft_uav.research.optimizer import select_constrained_configs
from raft_uav.research.tracklet_models import fit_platt_scaler, tracklet_feature_frame
from raft_uav.research.uncertainty import fit_conformal_radius


def test_factor_graph_smoother_returns_estimates() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "source": ["rf", "radar", "radar"],
        }
    )
    result = smooth_position_trajectory(measurements)
    assert result.success
    assert len(result.estimates) == 3
    assert np.isfinite(result.estimates[["east_m", "north_m", "up_m"]].to_numpy()).all()


def test_native_covariance_is_positive_semidefinite() -> None:
    covariance = enu_covariance_from_range_az_el(
        100.0,
        0.5,
        0.1,
        range_std_m=5.0,
        azimuth_std_rad=0.02,
        elevation_std_rad=0.03,
    )
    assert covariance.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_tracklet_features_and_platt_scaler() -> None:
    radar = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "frame_index": [0, 1, 0, 1],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "east_m": [0.0, 1.0, 100.0, 101.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "cat_prob_uav": [0.9, 0.8, 0.1, 0.2],
        }
    )
    features = tracklet_feature_frame(radar)
    assert len(features) == 2
    model = fit_platt_scaler([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    probabilities = model.predict_proba(pd.DataFrame({"logit_probability": [-3.0, 3.0]}))
    assert probabilities[1] > probabilities[0]


def test_conformal_and_constrained_optimizer() -> None:
    radius = fit_conformal_radius([1.0, 2.0, 3.0, 4.0], alpha=0.25)
    assert radius.radius_m >= 3.0
    rows = pd.DataFrame(
        {
            "method": ["a", "b"],
            "error_3d_rmse_m": [10.0, 8.0],
            "truth_coverage_rate": [0.9, 0.5],
        }
    )
    ranked = select_constrained_configs(
        rows,
        constraints={"truth_coverage_rate": (">=", 0.8)},
    )
    assert ranked.iloc[0]["method"] == "a"
    assert bool(ranked.iloc[0]["constraint_feasible"])
