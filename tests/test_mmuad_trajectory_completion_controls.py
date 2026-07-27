from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_gap_s", np.nan),
        ("fixed_lag_s", np.inf),
        ("smoothing_blend", np.nan),
        ("smoothing_blend", 1.01),
        ("speed_gate_mps", -1.0),
        ("outlier_replacement_max_gap_s", np.inf),
    ],
)
def test_completion_rejects_invalid_numeric_controls_before_empty_return(
    field: str,
    value: float,
) -> None:
    config = TrajectoryCompletionConfig(**{field: value})

    with pytest.raises(ValueError, match=field):
        complete_and_smooth_estimates(pd.DataFrame(), config=config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "unknown", "trajectory completion mode"),
        ("outlier_replacement", "unknown", "trajectory outlier replacement"),
    ],
)
def test_completion_rejects_unknown_modes_before_empty_return(
    field: str,
    value: str,
    message: str,
) -> None:
    config = TrajectoryCompletionConfig(**{field: value})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=message):
        complete_and_smooth_estimates(pd.DataFrame(), config=config)


def test_completion_preserves_default_empty_input_behavior() -> None:
    result = complete_and_smooth_estimates(pd.DataFrame())

    assert result.estimates.empty
    assert result.gap_summary.empty
    assert result.speed_gate_summary.empty
