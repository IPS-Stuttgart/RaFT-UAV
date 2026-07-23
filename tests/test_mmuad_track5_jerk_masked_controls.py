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
        ("max_jerk_mps3", np.ma.masked),
        ("smoothness_weight", np.ma.array(10.0, mask=True)),
        ("min_correction_m", np.ma.array(1.0, mask=True)),
        ("max_correction_m", np.ma.array(20.0, mask=True)),
        ("iterations", np.ma.array(2, mask=True)),
        ("repair_blend", np.ma.array(0.5, mask=True)),
    ],
)
def test_jerk_limit_rejects_masked_scalar_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=name):
        repair_track5_jerk_kinks(_submission(), **{name: value})
