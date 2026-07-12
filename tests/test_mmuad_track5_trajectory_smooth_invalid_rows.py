from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_smooth import smooth_track5_submission_rows


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 1.0, 2.0],
            "state_z_m": [1.0, 1.0, 1.0],
            "Classification": [1, 1, 1],
        }
    )


def test_trajectory_smoothing_rejects_nonfinite_rows() -> None:
    rows = _rows()
    rows.loc[1, "state_y_m"] = float("inf")

    with pytest.raises(ValueError, match="non-finite time/position values at rows \\[1\\]"):
        smooth_track5_submission_rows(rows)


def test_trajectory_smoothing_rejects_non_numeric_rows() -> None:
    rows = _rows()
    rows.loc[2, "time_s"] = "not-a-time"

    with pytest.raises(ValueError, match="non-finite time/position values at rows \\[2\\]"):
        smooth_track5_submission_rows(rows)
