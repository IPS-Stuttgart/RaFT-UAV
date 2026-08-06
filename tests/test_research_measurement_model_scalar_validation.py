from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.measurement_models import (
    covariance_columns_from_native_radar,
    enu_covariance_from_range_az_el,
    fit_linear_radar_bias_model,
    rf_quality_covariance_scale,
)


def _bias_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "range_m": [10.0, 20.0, 30.0],
            "residual_east_m": [1.0, 2.0, 3.0],
            "residual_north_m": [0.0, 0.5, 1.0],
            "residual_up_m": [0.0, 0.0, 0.5],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("range_m", True),
        ("azimuth_rad", np.array([0.1])),
        ("elevation_rad", 1.0 + 0.0j),
        ("range_std_m", np.nan),
        ("azimuth_std_rad", np.ma.masked),
        ("elevation_std_rad", -0.1),
        ("min_std_m", False),
    ],
)
def test_enu_covariance_rejects_invalid_scalar_controls(field: str, value: object) -> None:
    kwargs = {
        "range_m": 100.0,
        "azimuth_rad": 0.1,
        "elevation_rad": 0.2,
        "range_std_m": 5.0,
        "azimuth_std_rad": 0.01,
        "elevation_std_rad": 0.02,
        "min_std_m": 1.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        enu_covariance_from_range_az_el(**kwargs)


def test_enu_covariance_accepts_nested_zero_dimensional_real_scalars() -> None:
    boxed = np.empty((), dtype=object)
    boxed[()] = np.array(100.0)

    covariance = enu_covariance_from_range_az_el(
        boxed,
        np.array(0.1),
        np.float64(0.2),
        range_std_m=np.array(5.0),
        azimuth_std_rad=np.float64(0.01),
        elevation_std_rad=0.02,
        min_std_m=0.0,
    )

    assert np.isfinite(covariance).all()


@pytest.mark.parametrize(
    "ridge_lambda",
    [-1.0, np.nan, np.inf, True, np.array([1.0]), 1.0 + 0.0j],
)
def test_bias_fit_rejects_invalid_ridge_lambda(ridge_lambda: object) -> None:
    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_linear_radar_bias_model(
            _bias_examples(),
            feature_names=("intercept", "range_m"),
            ridge_lambda=ridge_lambda,
        )


def test_bias_fit_accepts_zero_dimensional_ridge_lambda() -> None:
    model = fit_linear_radar_bias_model(
        _bias_examples(),
        feature_names=("intercept", "range_m"),
        ridge_lambda=np.array(1.0),
    )

    assert np.isfinite(model.coefficients).all()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("range_std_m", np.nan),
        ("azimuth_std_deg", -1.0),
        ("elevation_std_deg", True),
        ("min_std_m", np.array([1.0])),
    ],
)
def test_covariance_columns_validate_controls_for_empty_frames(
    name: str,
    value: object,
) -> None:
    frame = pd.DataFrame(columns=["range_m", "azimuth_rad", "elevation_rad"])
    kwargs = {
        "range_std_m": 5.0,
        "azimuth_std_deg": 1.0,
        "elevation_std_deg": 1.0,
        "min_std_m": 1.0,
    }
    kwargs[name] = value

    with pytest.raises(ValueError, match=name):
        covariance_columns_from_native_radar(frame, **kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("base_scale", -1.0),
        ("base_scale", np.nan),
        ("base_scale", True),
        ("missing_penalty", -1.0),
        ("missing_penalty", np.inf),
        ("missing_penalty", np.array([2.0])),
    ],
)
def test_rf_quality_scale_rejects_invalid_controls_even_for_empty_frames(
    name: str,
    value: object,
) -> None:
    kwargs = {"base_scale": 1.0, "missing_penalty": 2.0}
    kwargs[name] = value

    with pytest.raises(ValueError, match=name):
        rf_quality_covariance_scale(pd.DataFrame(), **kwargs)
