from __future__ import annotations

import pandas as pd

from raft_uav.research.factor_graph import _radar_frame_groups


def test_factor_graph_grouping_falls_back_from_nonreal_time_to_frame_index() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0 + 2.0j, 2.0],
            "frame_index": [0, 1, 2],
            "track_id": [10, 11, 12],
        }
    )

    groups = _radar_frame_groups(radar)

    assert [key for key, _ in groups] == [
        ("frame_index_time", 0.0, 0.0),
        ("frame_index", 1.0),
        ("frame_index_time", 2.0, 2.0),
    ]
    assert [group["track_id"].tolist() for _, group in groups] == [[10], [11], [12]]


def test_factor_graph_grouping_falls_back_from_nonreal_index_to_time() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "frame_index": [0.0, 1.0 + 2.0j, 2.0],
            "track_id": [10, 11, 12],
        }
    )

    groups = _radar_frame_groups(radar)

    assert [key for key, _ in groups] == [
        ("frame_index_time", 0.0, 0.0),
        ("time_s", 1.0),
        ("frame_index_time", 2.0, 2.0),
    ]


def test_factor_graph_grouping_skips_only_rows_with_no_real_frame_key() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0 + 2.0j, 2.0],
            "frame_index": [0.0, 1.0 + 3.0j, 2.0],
            "track_id": [10, 11, 12],
        }
    )

    groups = _radar_frame_groups(radar)

    assert [group["track_id"].tolist() for _, group in groups] == [[10], [12]]
