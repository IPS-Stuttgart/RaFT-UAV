from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


@pytest.mark.parametrize(
    "value",
    [-1.0, np.nan, np.inf, True, 1 + 2j, np.array([1.0])],
)
def test_delayed_initialization_rejects_invalid_window(value: object) -> None:
    with pytest.raises(ValueError, match="window_s"):
        build_delayed_initial_hypotheses(
            rf_measurements=[],
            radar=pd.DataFrame(),
            window_s=value,
        )


@pytest.mark.parametrize(
    "value",
    [-1, 1.5, np.nan, True, 1 + 0j, np.array([2])],
)
def test_delayed_initialization_rejects_invalid_hypothesis_limit(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="max_hypotheses"):
        build_delayed_initial_hypotheses(
            rf_measurements=[],
            radar=pd.DataFrame(),
            max_hypotheses=value,
        )


@pytest.mark.parametrize(
    "field",
    ["initial_position_std_m", "initial_velocity_std_mps"],
)
@pytest.mark.parametrize(
    "value",
    [-1.0, np.nan, np.inf, True, np.array([1.0])],
)
def test_delayed_initialization_rejects_invalid_standard_deviations(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        build_delayed_initial_hypotheses(
            rf_measurements=[],
            radar=pd.DataFrame(),
            **{field: value},
        )


def test_delayed_initialization_preserves_valid_scalar_like_controls() -> None:
    rf = [SimpleNamespace(time_s=0.0, vector=np.array([1.0, 2.0, 3.0]))]

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=rf,
        radar=pd.DataFrame(),
        window_s=np.array(0.0),
        max_hypotheses="1",
        initial_position_std_m=np.array(2.0),
        initial_velocity_std_mps="3.0",
    )

    assert len(hypotheses) == 1
    np.testing.assert_allclose(
        np.diag(hypotheses[0].covariance),
        [4.0, 4.0, 4.0, 9.0, 9.0, 9.0],
    )
