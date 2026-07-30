from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _submission() -> pd.DataFrame:
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
    ("column", "value"),
    [
        ("time_s", 1.0 + 2.0j),
        ("state_x_m", np.complex128(10.0 + 3.0j)),
        ("state_y_m", np.array(1.0 + 4.0j)),
        ("state_z_m", np.complex128(0.0 + 0.0j)),
    ],
)
def test_acceleration_limit_rejects_complex_trajectory_cells(
    column: str,
    value: object,
) -> None:
    rows = _submission()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value

    with pytest.raises(ValueError, match="complex numeric values") as error:
        repair_track5_acceleration_kinks(rows)

    assert f"{column} rows [1]" in str(error.value)
