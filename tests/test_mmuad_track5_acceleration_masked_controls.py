from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _kink_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_acceleration_mps2", np.ma.array(5.0, mask=True)),
        ("max_direct_speed_mps", np.ma.array(20.0, mask=True)),
        ("min_interpolation_residual_m", np.ma.array(1.0, mask=True)),
        ("repair_blend", np.ma.array(0.5, mask=True)),
        ("iterations", np.ma.array(1, mask=True)),
    ],
)
def test_acceleration_limit_rejects_masked_scalar_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_acceleration_kinks(_kink_submission(), **{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "max_acceleration_mps2",
        "max_direct_speed_mps",
        "min_interpolation_residual_m",
        "repair_blend",
        "iterations",
    ],
)
def test_acceleration_limit_rejects_masked_constant_controls(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_acceleration_kinks(
            _kink_submission(),
            **{name: np.ma.masked},
        )
