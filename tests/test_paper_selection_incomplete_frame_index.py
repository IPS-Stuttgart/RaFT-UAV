from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.paper_selection import select_paper_strict_raw_radar_track


def test_partial_frame_indices_fall_back_to_time_for_segment_selection() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 100.0, 101.0, 10.0, 10.5],
            "frame_index": [0.0, np.nan, 1.0, 0.0, 1.0],
            "track_id": [1, 1, 1, 2, 2],
            "east_m": [0.0, 100.0, 101.0, 10.0, 10.5],
            "north_m": [0.0, 0.0, 0.0, 1.0, 1.0],
            "up_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "range_m": [100.0, 100.0, 100.0, 100.0, 100.0],
            "cat_prob_uav": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["track_id"].astype(int).unique().tolist() == [1]
    assert selected["time_s"].tolist() == [100.0, 101.0]
