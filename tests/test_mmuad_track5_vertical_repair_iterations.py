from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 100.0, 2.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    "iterations",
    [0, -1, 1.5, True, False, np.nan, np.inf, -np.inf],
)
def test_vertical_repair_rejects_invalid_iteration_counts(iterations: object) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        repair_track5_vertical_spikes(_submission(), iterations=iterations)


def test_vertical_repair_accepts_integer_equivalent_iteration_count() -> None:
    expected, expected_diagnostics = repair_track5_vertical_spikes(
        _submission(),
        iterations=1,
    )
    actual, actual_diagnostics = repair_track5_vertical_spikes(
        _submission(),
        iterations=1.0,
    )

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_diagnostics, expected_diagnostics)
