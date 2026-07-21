from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.diagnostics import latency_curve

_LATENCY_COLUMNS = [
    "latency_s",
    "truth_rows",
    "covered_truth_rows",
    "truth_coverage_rate",
    "error_3d_count",
    "error_3d_rmse_m",
    "error_3d_p95_m",
]


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def test_latency_curve_returns_stable_empty_schema() -> None:
    result = latency_curve({}, _truth())

    assert result.empty
    assert result.columns.tolist() == _LATENCY_COLUMNS


@pytest.mark.parametrize(
    "gate",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
    ],
)
def test_latency_curve_rejects_invalid_time_gate(gate: object) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        latency_curve({}, _truth(), max_time_delta_s=gate)


def test_latency_curve_preserves_valid_zero_gate() -> None:
    truth = _truth()

    result = latency_curve(
        {0.0: truth.copy()},
        truth,
        max_time_delta_s=np.array(0.0),
    )

    assert result.loc[0, "covered_truth_rows"] == len(truth)
    assert result.loc[0, "truth_coverage_rate"] == 1.0
    assert result.loc[0, "error_3d_rmse_m"] == 0.0
