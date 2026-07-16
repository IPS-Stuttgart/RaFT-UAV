from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_table import (
    _interpolate_selected_radar_to_frame_times,
)


def test_paper_table_interpolation_ignores_nonfinite_anchor_rows() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 1, 2],
            "time_s": [0.0, 1.0, 2.0],
            "track_id": [7, 8, 7],
            "east_m": [0.0, 100.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )
    selected = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, np.nan],
            "track_id": [7, 7, 7, 7],
            "east_m": [0.0, np.nan, 20.0, 999.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
        }
    )

    interpolated = _interpolate_selected_radar_to_frame_times(
        radar,
        selected,
        association_mode="test-interpolation",
    )

    assert interpolated["time_s"].tolist() == [0.0, 1.0, 2.0]
    np.testing.assert_allclose(interpolated["east_m"], [0.0, 10.0, 20.0])
    assert interpolated["association_anchor_count"].tolist() == [2, 2, 2]
    assert interpolated["track_id"].tolist() == [7, 7, 7]
