from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 1.0, 30.0, 3.0, 4.0, 5.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [1.0] * 6,
            "Classification": [2] * 6,
        }
    )


@pytest.mark.parametrize(
    "bad_iterations",
    [
        0,
        -1,
        1.5,
        True,
        np.bool_(True),
        np.nan,
        np.inf,
        -np.inf,
        pd.NA,
        np.array([1]),
    ],
)
def test_jerk_limit_rejects_invalid_iterations(bad_iterations: object) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive finite integer"):
        repair_track5_jerk_kinks(_submission(), iterations=bad_iterations)


@pytest.mark.parametrize(
    "iterations",
    [1, 1.0, np.int64(1), np.float64(1.0), np.array(1)],
)
def test_jerk_limit_accepts_integer_equivalent_iterations(iterations: object) -> None:
    repaired, diagnostics = repair_track5_jerk_kinks(
        _submission(),
        max_jerk_mps3=5.0,
        smoothness_weight=100.0,
        min_correction_m=1.0,
        iterations=iterations,
    )

    assert len(repaired) == len(_submission())
    assert len(diagnostics) == len(_submission())
