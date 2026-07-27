from __future__ import annotations

import pandas as pd

from raft_uav.paper_selection import select_paper_strict_raw_radar_track


def test_reused_frame_indices_do_not_merge_disconnected_track_epochs() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 10.0, 11.0],
            "frame_index": [0, 1, 0, 1],
            "track_id": [7, 7, 7, 7],
            "track_index": [0, 1, 2, 3],
            "cat_prob_uav": [0.9, 0.9, 0.1, 0.1],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 1.0]
    assert selected["frame_index"].tolist() == [0, 1]


def test_repeated_single_frame_at_later_time_starts_new_epoch() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 10.0],
            "frame_index": [0, 0],
            "track_id": [7, 7],
            "track_index": [0, 1],
            "cat_prob_uav": [0.9, 0.1],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0]


def test_duplicate_rows_from_same_physical_frame_remain_together() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0],
            "frame_index": [0, 0, 1],
            "track_id": [7, 7, 7],
            "track_index": [0, 1, 2],
            "cat_prob_uav": [0.5, 0.5, 0.5],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["time_s"].tolist() == [0.0, 0.0, 1.0]
