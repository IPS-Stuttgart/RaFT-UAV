from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import smooth_position_trajectory


def test_factor_graph_skips_rows_without_finite_time_and_position() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, np.inf, 2.0],
            "east_m": [0.0, np.nan, 100.0, 2.0],
            "north_m": [0.0, np.nan, 100.0, 0.0],
            "up_m": [0.0, np.nan, 100.0, 0.0],
        }
    )

    result = smooth_position_trajectory(measurements)

    assert result.success
    np.testing.assert_allclose(result.estimates["time_s"], [0.0, 2.0])
    np.testing.assert_allclose(
        result.estimates[["east_m", "north_m", "up_m"]],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )


def test_factor_graph_rejects_measurements_without_any_usable_row() -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, np.inf],
            "east_m": [np.nan, 1.0],
            "north_m": [np.nan, 1.0],
            "up_m": [np.nan, 1.0],
        }
    )

    with pytest.raises(ValueError, match="no finite time/position rows"):
        smooth_position_trajectory(measurements)
