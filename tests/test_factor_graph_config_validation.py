from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    smooth_position_trajectory,
)


def _measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "source": ["rf", "radar", "radar"],
        }
    )


def _config(**overrides: object) -> LeastSquaresSmoothingConfig:
    values: dict[str, object] = {
        "motion_std_mps2": 4.0,
        "measurement_std_m": 25.0,
        "rf_std_m": 50.0,
        "robust_loss": "soft_l1",
        "max_nfev": 200,
    }
    values.update(overrides)
    return LeastSquaresSmoothingConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("motion_std_mps2", -1.0),
        ("motion_std_mps2", np.ma.array(4.0, mask=True)),
        ("measurement_std_m", 0.0),
        ("measurement_std_m", -25.0),
        ("rf_std_m", 0.0),
        ("rf_std_m", np.inf),
        ("max_nfev", 0),
        ("max_nfev", 1.5),
        ("max_nfev", True),
    ],
)
def test_smoother_rejects_invalid_numeric_controls(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        smooth_position_trajectory(_measurements(), config=_config(**{field: value}))


def test_smoother_validates_controls_before_empty_input_return() -> None:
    empty = pd.DataFrame(columns=["time_s", "east_m", "north_m", "up_m"])

    with pytest.raises(ValueError, match="measurement_std_m"):
        smooth_position_trajectory(empty, config=_config(measurement_std_m=0.0))


def test_smoother_accepts_exact_scalar_like_controls() -> None:
    result = smooth_position_trajectory(
        _measurements(),
        config=_config(
            motion_std_mps2=np.array(0.0),
            measurement_std_m=np.float64(25.0),
            rf_std_m=np.array(50.0),
            max_nfev=np.int64(20),
        ),
    )

    assert result.success
    assert result.iterations <= 20
