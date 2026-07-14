from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.paper_selection import select_paper_strict_raw_radar_track


def test_partial_frame_indices_fall_back_to_time_continuity() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 10.0],
            "frame_index": [0.0, np.nan, 2.0],
            "track_id": [7, 7, 7],
            "track_index": [0, 1, 2],
            "cat_prob_uav": [0.5, 0.5, 0.5],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 1.0]


def test_nonfinite_frame_indices_fall_back_to_time_continuity() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 10.0],
            "frame_index": [0.0, np.inf, 2.0],
            "track_id": [7, 7, 7],
            "track_index": [0, 1, 2],
            "cat_prob_uav": [0.5, 0.5, 0.5],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 1.0]


def test_complete_frame_indices_remain_authoritative() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 100.0, 101.0],
            "frame_index": [0, 1, 10],
            "track_id": [7, 7, 7],
            "track_index": [0, 1, 2],
            "cat_prob_uav": [0.5, 0.5, 0.5],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 100.0]
