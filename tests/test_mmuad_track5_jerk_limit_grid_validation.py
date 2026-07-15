from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_jerk_limit import repair_track5_jerk_kinks


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 6,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "state_x_m": [0.0, 1.0, 30.0, 3.0, 4.0, 5.0],
            "state_y_m": [0.0] * 6,
            "state_z_m": [5.0] * 6,
            "Classification": [2] * 6,
        }
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", "not-a-time"),
        ("state_x_m", np.nan),
        ("state_y_m", np.inf),
        ("state_z_m", -np.inf),
    ],
)
def test_jerk_limit_rejects_malformed_grid_rows(
    column: str,
    value: object,
) -> None:
    submission = _submission_rows()
    submission.loc[2, column] = value

    with pytest.raises(
        ValueError,
        match=r"non-finite time or position values at row indices: 2",
    ):
        repair_track5_jerk_kinks(submission)
