from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
            "Classification": [2, 2],
        },
        index=[10, 20],
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", math.nan),
        ("state_x_m", math.inf),
    ],
)
def test_speed_limit_rejects_invalid_rows_instead_of_dropping_them(
    column: str,
    value: float,
) -> None:
    submission = _submission_rows()
    submission.loc[20, column] = value

    with pytest.raises(
        ValueError,
        match=r"non-finite time or position values at row indices: 20",
    ):
        project_track5_speed_limit(submission, max_speed_mps=10.0)
