import numpy as np
import pandas as pd

from raft_uav.research.measurement_models import fit_linear_radar_bias_model


def test_ridge_exemption_follows_named_intercept_in_custom_feature_order() -> None:
    examples = pd.DataFrame(
        {
            "range_m": [1.0, 2.0, 3.0, 4.0],
            "residual_east_m": [10.0, 10.0, 10.0, 10.0],
            "residual_north_m": [-5.0, -5.0, -5.0, -5.0],
            "residual_up_m": [2.0, 2.0, 2.0, 2.0],
        }
    )

    model = fit_linear_radar_bias_model(
        examples,
        feature_names=("range_m", "intercept"),
        ridge_lambda=100.0,
    )

    assert model.feature_names == ("range_m", "intercept")
    np.testing.assert_allclose(model.coefficients[0], np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(model.coefficients[1], np.array([10.0, -5.0, 2.0]), atol=1.0e-12)
