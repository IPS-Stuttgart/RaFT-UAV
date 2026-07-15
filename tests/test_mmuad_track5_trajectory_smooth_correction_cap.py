from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_smooth import smooth_track5_submission_rows


def _spiked_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [0.0, 1.0, 12.0, 3.0, 4.0],
            "state_y_m": [0.0, 0.0, 10.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 11.0, 1.0, 1.0],
            "Classification": [2] * 5,
        }
    )


def test_zero_correction_cap_preserves_input_trajectory() -> None:
    rows = _spiked_rows()

    smoothed, diagnostics = smooth_track5_submission_rows(
        rows,
        window_s=2.0,
        bandwidth_s=1.0,
        blend=1.0,
        max_correction_m=0.0,
        min_neighbors=3,
    )

    coordinates = ["state_x_m", "state_y_m", "state_z_m"]
    np.testing.assert_allclose(smoothed[coordinates], rows[coordinates])
    assert float(diagnostics["raw_correction_m"].max()) > 0.0
    assert diagnostics["applied_correction_m"].eq(0.0).all()
    assert diagnostics["capped"].any()


@pytest.mark.parametrize(
    "max_correction_m",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_correction_caps_are_rejected(max_correction_m: float) -> None:
    with pytest.raises(ValueError, match="max_correction_m"):
        smooth_track5_submission_rows(
            _spiked_rows(),
            max_correction_m=max_correction_m,
        )
