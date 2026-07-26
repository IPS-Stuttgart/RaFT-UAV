from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import (
    repair_track5_acceleration_kinks,
)
from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes
from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks
from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 1.0, 2.0, 3.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
            "Classification": [2, 2, 2, 2],
        }
    )


_REPAIRS: tuple[Callable[..., object], ...] = (
    repair_track5_acceleration_kinks,
    repair_track5_hampel_spikes,
    repair_track5_jerk_kinks,
    repair_track5_vertical_spikes,
)


@pytest.mark.parametrize(
    "repair",
    _REPAIRS,
    ids=("acceleration", "hampel", "jerk", "vertical"),
)
@pytest.mark.parametrize(
    "value",
    [
        1.0 + 2.0j,
        np.complex128(3.0 + 0.0j),
        np.array(4.0 + 5.0j),
        np.ma.array(6.0 + 0.0j, mask=False),
    ],
)
def test_track5_repairs_reject_complex_coordinate_cells(
    repair: Callable[..., object],
    value: object,
) -> None:
    rows = _submission()
    rows["state_x_m"] = rows["state_x_m"].astype(object)
    rows.at[1, "state_x_m"] = value

    with pytest.raises(
        ValueError,
        match=r"complex numeric values: state_x_m rows \[1\]",
    ):
        repair(rows)
