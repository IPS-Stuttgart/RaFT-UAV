from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.uncertainty import VarianceHead, feature_matrix


def _radar_velocity_frame() -> pd.DataFrame:
    limit = np.finfo(float).max
    return pd.DataFrame(
        {
            "velocity_east_mps": [3.0, limit, limit],
            "velocity_north_mps": [4.0, 0.0, limit],
            "velocity_down_mps": [12.0, 0.0, limit],
        }
    )


def test_radar_velocity_norm_is_finite_for_finite_components() -> None:
    limit = np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        features = feature_matrix(
            _radar_velocity_frame(),
            "radar",
            ("velocity_norm",),
        )

    assert features.shape == (3, 1)
    assert features[0, 0] == 13.0
    assert features[1, 0] == limit
    assert features[2, 0] == limit
    assert np.isfinite(features).all()


def test_zero_weight_extreme_velocity_does_not_poison_variance_prediction() -> None:
    head = VarianceHead(
        source="radar",
        dimension="east",
        feature_names=("intercept", "velocity_norm"),
        coefficients=(0.0, 0.0),
        min_std_m=3.0,
        max_std_m=300.0,
        training_rows=1,
    )

    with np.errstate(over="raise", invalid="raise"):
        prediction = head.predict(_radar_velocity_frame().iloc[[2]])

    np.testing.assert_allclose(prediction, [9.0])
    assert np.isfinite(prediction).all()
