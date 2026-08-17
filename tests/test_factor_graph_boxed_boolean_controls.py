from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    coordinate_descent_association_and_smoothing,
    smooth_position_trajectory,
)


def _measurements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
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
        ("motion_std_mps2", np.array(False)),
        ("measurement_std_m", np.array(True)),
        ("rf_std_m", np.array(True, dtype=object)),
        ("max_nfev", np.array(True)),
    ],
)
def test_smoother_rejects_boxed_boolean_numeric_controls(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        smooth_position_trajectory(
            _measurements(),
            config=_config(**{field: value}),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("iterations", np.array(True)),
        ("candidate_gate_m", np.array(False, dtype=object)),
    ],
)
def test_coordinate_descent_rejects_boxed_boolean_controls(
    field: str,
    value: object,
) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match=field):
        coordinate_descent_association_and_smoothing(
            _measurements(),
            **kwargs,  # type: ignore[arg-type]
        )
