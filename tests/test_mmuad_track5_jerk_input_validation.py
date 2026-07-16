from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 1.0, 30.0, 3.0, 4.0, 5.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [1.0] * 6,
            "Classification": [2] * 6,
        }
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_jerk_mps3", True),
        ("max_jerk_mps3", np.bool_(True)),
        ("smoothness_weight", False),
        ("min_correction_m", True),
        ("max_correction_m", True),
        ("repair_blend", True),
        ("repair_blend", np.array(False)),
    ],
)
def test_jerk_limit_rejects_boolean_controls(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_jerk_kinks(_submission(), **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_jerk_mps3", np.array([5.0])),
        ("smoothness_weight", np.array([10.0])),
        ("min_correction_m", np.array([1.0])),
        ("max_correction_m", np.array([20.0])),
        ("repair_blend", np.array([0.5])),
    ],
)
def test_jerk_limit_rejects_non_scalar_controls(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_jerk_kinks(_submission(), **{name: value})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_jerk_mps3", np.nan),
        ("smoothness_weight", np.inf),
        ("min_correction_m", -np.inf),
        ("max_correction_m", np.nan),
        ("repair_blend", np.inf),
    ],
)
def test_jerk_limit_rejects_non_finite_controls(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_jerk_kinks(_submission(), **{name: value})


def test_jerk_limit_accepts_numeric_scalar_like_controls() -> None:
    repaired, diagnostics = repair_track5_jerk_kinks(
        _submission(),
        max_jerk_mps3=np.array(5.0),
        smoothness_weight=np.float64(100.0),
        min_correction_m=np.int64(1),
        max_correction_m=np.array(20.0),
        iterations=np.array(1),
        repair_blend=np.float64(0.5),
    )

    assert len(repaired) == len(_submission())
    assert len(diagnostics) == len(_submission())
