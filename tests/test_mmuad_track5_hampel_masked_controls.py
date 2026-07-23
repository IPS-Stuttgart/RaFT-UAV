from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 20.0, 3.0, 4.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [1.0] * 5,
            "Classification": [2] * 5,
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_radius", np.ma.array(3, mask=True)),
        ("iterations", np.ma.array(2, mask=True)),
        ("sigma_threshold", np.ma.array(4.0, mask=True)),
        ("min_scale_m", np.ma.array(1.5, mask=True)),
        ("min_residual_m", np.ma.array(2.5, mask=True)),
        ("repair_blend", np.ma.array(0.25, mask=True)),
    ],
)
def test_hampel_repair_rejects_masked_scalar_controls(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        repair_track5_hampel_spikes(_submission(), **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_radius", np.ma.masked),
        ("sigma_threshold", np.ma.masked),
    ],
)
def test_hampel_repair_rejects_masked_singleton_controls(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        repair_track5_hampel_spikes(_submission(), **{field: value})
