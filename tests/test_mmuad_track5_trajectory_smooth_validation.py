from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_smooth import smooth_track5_submission_rows


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"blend": math.nan}, "blend"),
        ({"blend": math.inf}, "blend"),
        ({"window_s": math.nan}, "window_s"),
        ({"window_s": math.inf}, "window_s"),
        ({"bandwidth_s": math.nan}, "bandwidth_s"),
        ({"bandwidth_s": math.inf}, "bandwidth_s"),
        ({"max_correction_m": math.nan}, "max_correction_m"),
        ({"max_correction_m": math.inf}, "max_correction_m"),
        ({"max_correction_m": -1.0}, "max_correction_m"),
        ({"min_neighbors": 0}, "min_neighbors"),
        ({"min_neighbors": -2}, "min_neighbors"),
    ],
)
def test_smoother_rejects_invalid_control_parameters(
    kwargs: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_track5_submission_rows(_rows(), **kwargs)


def test_smoother_still_accepts_disabled_correction_cap() -> None:
    smoothed, diagnostics = smooth_track5_submission_rows(_rows(), max_correction_m=None)

    assert len(smoothed) == 3
    assert len(diagnostics) == 3
