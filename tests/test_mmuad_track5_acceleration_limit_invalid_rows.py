from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("time_s", [0.0, "invalid", 2.0]),
        ("state_x_m", [0.0, float("nan"), 2.0]),
    ],
)
def test_acceleration_limit_rejects_invalid_rows_instead_of_dropping_them(
    column: str,
    values: list[object],
) -> None:
    submission = _submission()
    submission[column] = values

    with pytest.raises(ValueError, match=r"row indices: 1$"):
        repair_track5_acceleration_kinks(submission)
