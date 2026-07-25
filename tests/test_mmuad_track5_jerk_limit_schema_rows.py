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
        },
        index=[10, 11, 12, 13, 14, 15],
    )


@pytest.mark.parametrize("value", [None, np.nan, "", "   "])
def test_jerk_limit_rejects_invalid_sequence_identifiers(value: object) -> None:
    submission = _submission_rows()
    submission["sequence_id"] = submission["sequence_id"].astype(object)
    submission.loc[12, "sequence_id"] = value

    with pytest.raises(
        ValueError,
        match=r"invalid sequence identifiers at row indices: 12",
    ):
        repair_track5_jerk_kinks(submission)


@pytest.mark.parametrize("value", [None, np.nan, "not-a-class", True, 99])
def test_jerk_limit_rejects_invalid_classification_values(value: object) -> None:
    submission = _submission_rows()
    submission["Classification"] = submission["Classification"].astype(object)
    submission.loc[12, "Classification"] = value

    with pytest.raises(
        ValueError,
        match=r"invalid Classification values at row indices: 12",
    ):
        repair_track5_jerk_kinks(submission)
