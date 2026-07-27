from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    smooth_position_trajectory,
)


def _measurements(*, empty: bool) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    return rows.iloc[0:0].copy() if empty else rows


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("robust_loss", ["", "typo", None, 1])
def test_smoother_rejects_invalid_robust_loss_before_data_dependent_return(
    empty: bool,
    robust_loss: object,
) -> None:
    config = LeastSquaresSmoothingConfig(robust_loss=robust_loss)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="robust_loss"):
        smooth_position_trajectory(_measurements(empty=empty), config=config)


@pytest.mark.parametrize("robust_loss", ["linear", "huber", "soft_l1", "cauchy", "arctan"])
def test_smoother_accepts_supported_robust_loss_names(robust_loss: str) -> None:
    result = smooth_position_trajectory(
        _measurements(empty=True),
        config=LeastSquaresSmoothingConfig(robust_loss=robust_loss),
    )

    assert result.success
    assert result.message == "empty"


def test_smoother_preserves_scipy_callable_loss_support() -> None:
    def loss(z: object) -> object:
        return z

    callable_loss: Callable[[object], object] = loss
    result = smooth_position_trajectory(
        _measurements(empty=True),
        config=LeastSquaresSmoothingConfig(robust_loss=callable_loss),  # type: ignore[arg-type]
    )

    assert result.success
    assert result.message == "empty"
