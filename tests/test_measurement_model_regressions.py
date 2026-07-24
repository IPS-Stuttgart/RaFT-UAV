import numpy as np
import pandas as pd
import pytest

from raft_uav.research.measurement_models import (
    covariance_columns_from_native_radar,
    enu_covariance_from_range_az_el,
    fit_linear_radar_bias_model,
)


@pytest.mark.parametrize(
    "field",
    [
        "range_m",
        "azimuth_rad",
        "elevation_rad",
        "range_std_m",
        "azimuth_std_rad",
        "elevation_std_rad",
        "min_std_m",
    ],
)
@pytest.mark.parametrize("value", [True, np.array([1.0])])
def test_native_covariance_rejects_non_scalar_or_boolean_values(field, value):
    params = {
        "range_m": 100.0,
        "azimuth_rad": 0.5,
        "elevation_rad": 0.1,
        "range_std_m": 5.0,
        "azimuth_std_rad": 0.02,
        "elevation_std_rad": 0.03,
        "min_std_m": 1.0,
    }
    params[field] = value
    with pytest.raises(ValueError, match=field):
        enu_covariance_from_range_az_el(**params)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("range_std_m", True),
        ("azimuth_std_deg", np.inf),
        ("elevation_std_deg", np.array([1.0])),
        ("min_std_m", -1.0),
    ],
)
def test_covariance_columns_validates_controls_for_empty_frames(field, value):
    frame = pd.DataFrame(columns=["range_m", "azimuth_rad", "elevation_rad"])
    params = {
        "range_std_m": 5.0,
        "azimuth_std_deg": 1.0,
        "elevation_std_deg": 1.0,
        "min_std_m": 1.0,
    }
    params[field] = value
    with pytest.raises(ValueError, match=field):
        covariance_columns_from_native_radar(frame, **params)


def test_bias_model_regularizes_first_feature_when_it_is_not_intercept():
    examples = pd.DataFrame(
        {
            "range_m": [1.0, 2.0],
            "residual_east_m": [1.0, 2.0],
            "residual_north_m": [2.0, 4.0],
            "residual_up_m": [3.0, 6.0],
        }
    )
    model = fit_linear_radar_bias_model(
        examples,
        feature_names=("range_m",),
        ridge_lambda=1.0,
    )
    np.testing.assert_allclose(model.coefficients[:, 0], [5.0 / 6.0])
    np.testing.assert_allclose(model.coefficients[:, 1], [10.0 / 6.0])
    np.testing.assert_allclose(model.coefficients[:, 2], [15.0 / 6.0])


@pytest.mark.parametrize("value", [-1.0, np.nan, True, np.array([1.0])])
def test_bias_model_rejects_invalid_ridge_lambda(value):
    examples = pd.DataFrame(
        {
            "range_m": [1.0],
            "residual_east_m": [1.0],
            "residual_north_m": [1.0],
            "residual_up_m": [1.0],
        }
    )
    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_linear_radar_bias_model(
            examples,
            feature_names=("range_m",),
            ridge_lambda=value,
        )


def test_bias_model_rejects_empty_feature_set():
    examples = pd.DataFrame(
        {
            "residual_east_m": [1.0],
            "residual_north_m": [1.0],
            "residual_up_m": [1.0],
        }
    )
    with pytest.raises(ValueError, match="feature_names"):
        fit_linear_radar_bias_model(examples, feature_names=())
