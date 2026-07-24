from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 100.0, 200.0, 300.0],
            "state_y_m": [0.0] * 4,
            "state_z_m": [5.0] * 4,
            "Classification": [2] * 4,
        },
        index=[10, 11, 12, 13],
    )


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("time_s", "not-a-time"),
        ("state_x_m", np.nan),
        ("state_y_m", np.inf),
        ("state_z_m", -np.inf),
    ],
)
def test_speed_limit_rejects_malformed_grid_rows(
    column: str,
    invalid_value: object,
) -> None:
    submission = _submission_rows()
    submission[column] = submission[column].astype(object)
    submission.loc[12, column] = invalid_value

    with pytest.raises(
        ValueError,
        match=r"non-finite or non-numeric time or position values at row indices: 12",
    ):
        project_track5_speed_limit(submission, max_speed_mps=10.0)


def test_speed_limit_preserves_every_valid_grid_row() -> None:
    submission = _submission_rows()

    limited, diagnostics = project_track5_speed_limit(
        submission,
        max_speed_mps=10.0,
    )

    assert len(limited) == len(submission)
    assert len(diagnostics) == len(submission)
    assert limited["time_s"].tolist() == submission["time_s"].tolist()
