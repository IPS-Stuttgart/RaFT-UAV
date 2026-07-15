from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_regularizer import regularize_track5_estimates


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 100.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("smoothness_weight", np.nan),
        ("smoothness_weight", np.inf),
        ("smoothness_weight", True),
        ("smoothness_weight", np.array([1.0])),
        ("huber_delta_m", np.nan),
        ("huber_delta_m", np.inf),
        ("huber_delta_m", True),
        ("huber_delta_m", np.array([1.0])),
        ("observation_sigma_m", np.nan),
        ("observation_sigma_m", np.inf),
        ("observation_sigma_m", True),
        ("observation_sigma_m", np.array([1.0])),
        ("iterations", np.nan),
        ("iterations", np.inf),
        ("iterations", 1.5),
        ("iterations", True),
        ("iterations", np.array([1])),
    ],
)
def test_regularizer_rejects_invalid_controls(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        regularize_track5_estimates(_estimates(), **{field: value})


def test_regularizer_accepts_finite_numpy_scalars() -> None:
    regularized, diagnostics = regularize_track5_estimates(
        _estimates(),
        smoothness_weight=np.array(100.0),
        huber_delta_m=np.float64(10.0),
        iterations=np.int64(2),
        observation_sigma_m=np.float32(1.0),
    )

    coordinates = regularized[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    assert np.isfinite(coordinates).all()
    assert diagnostics.loc[0, "iterations"] == 2
    assert diagnostics.loc[0, "smoothness_weight"] == pytest.approx(100.0)
