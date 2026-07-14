from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes


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
        ("state_z_m", -math.inf),
        ("state_y_m", "not-a-number"),
    ],
)
def test_hampel_repair_rejects_invalid_rows_instead_of_dropping_them(
    column: str,
    value: object,
) -> None:
    submission = _submission_rows()
    if isinstance(value, str):
        submission[column] = submission[column].astype(object)
    submission.loc[20, column] = value

    with pytest.raises(
        ValueError,
        match=r"non-finite time or position values at row indices: 20",
    ):
        repair_track5_hampel_spikes(submission)


def test_hampel_repair_preserves_all_valid_grid_rows() -> None:
    repaired, diagnostics = repair_track5_hampel_spikes(_submission_rows())

    assert repaired["time_s"].tolist() == [0.0, 1.0]
    assert len(diagnostics) == 2
