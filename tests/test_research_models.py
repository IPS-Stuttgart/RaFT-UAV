from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import (
    coordinate_descent_association_and_smoothing,
    smooth_position_trajectory,
)
from raft_uav.research.measurement_models import enu_covariance_from_range_az_el
from raft_uav.research.optimizer import select_constrained_configs
from raft_uav.research.runtime_modes import backward_repair_associations
from raft_uav.research.tracklet_models import (
    StandardizedLogisticModel,
    fit_platt_scaler,
    tracklet_feature_frame,
)
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


def test_coordinate_descent_skips_all_invalid_candidate_frame() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "track_id": [1, 2],
            "time_s": [0.0, 1.0],
            "east_m": [0.0, np.nan],
            "north_m": [0.0, np.nan],
            "up_m": [0.0, np.nan],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    _, selected = coordinate_descent_association_and_smoothing(
        radar,
        rf,
        iterations=1,
        candidate_gate_m=10.0,
    )

    assert selected["frame_index"].tolist() == [0]


def test_backward_repair_skips_all_invalid_candidate_frame() -> None:
    selected = pd.DataFrame(
        {
            "frame_index": [0, 2],
            "track_id": [1, 1],
            "time_s": [0.0, 2.0],
            "east_m": [0.0, 2.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    candidates = pd.DataFrame(
        {
            "frame_index": [0, 1, 2],
            "track_id": [1, 99, 1],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, np.nan, 2.0],
            "north_m": [0.0, np.nan, 0.0],
            "up_m": [0.0, np.nan, 0.0],
        }
    )

    repaired = backward_repair_associations(
        selected,
        candidates,
        max_gap_s=3.0,
        max_repair_distance_m=10.0,
    )

    assert repaired["frame_index"].tolist() == [0, 2]


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


def test_standardized_logistic_probabilities_are_stable_for_extreme_logits() -> None:
    model = StandardizedLogisticModel(
        feature_names=("x",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([1.0]),
        intercept=0.0,
    )
    features = pd.DataFrame({"x": [-1000.0, 0.0, 1000.0]})

    with np.errstate(over="raise", invalid="raise", under="ignore"):
        probabilities = model.predict_proba(features)

    np.testing.assert_allclose(
        probabilities,
        np.array([0.0, 0.5, 1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_standardized_logistic_model_save_round_trips_strict_json(tmp_path: Path) -> None:
    model = StandardizedLogisticModel(
        feature_names=("x",),
        mean=np.array([0.0]),
        scale=np.array([0.0]),
        weights=np.array([1.0]),
        intercept=0.0,
    )
    path = tmp_path / "nested" / "tracklet_model.json"

    model.save(path)

    payload_text = path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    loaded = StandardizedLogisticModel.load(path)

    assert "NaN" not in payload_text
    assert payload["scale"] == [1.0]
    np.testing.assert_allclose(
        loaded.predict_proba(pd.DataFrame({"x": [0.0]})),
        np.array([0.5]),
    )


def test_standardized_logistic_model_rejects_non_finite_parameters() -> None:
    with pytest.raises(ValueError, match="mean"):
        StandardizedLogisticModel(
            feature_names=("x",),
            mean=np.array([np.nan]),
            scale=np.array([1.0]),
            weights=np.array([1.0]),
            intercept=0.0,
        )

    with pytest.raises(ValueError, match="intercept"):
        StandardizedLogisticModel(
            feature_names=("x",),
            mean=np.array([0.0]),
            scale=np.array([1.0]),
            weights=np.array([1.0]),
            intercept=float("nan"),
        )


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
