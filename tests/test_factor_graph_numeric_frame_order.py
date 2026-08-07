from __future__ import annotations

import pandas as pd

from raft_uav.research.factor_graph import (
    _radar_frame_groups,
    coordinate_descent_association_and_smoothing,
)


def test_factor_graph_grouping_sorts_numeric_string_keys_chronologically() -> None:
    numeric = pd.DataFrame(
        {
            "time_s": [1.0, 10.0, 2.0],
            "frame_index": [1, 10, 2],
            "track_id": [101, 110, 102],
        }
    )
    strings = numeric.assign(
        time_s=numeric["time_s"].astype(str),
        frame_index=numeric["frame_index"].astype(str),
    )

    expected = _radar_frame_groups(numeric)
    actual = _radar_frame_groups(strings)

    expected_keys = [
        ("frame_index_time", 1.0, 1.0),
        ("frame_index_time", 2.0, 2.0),
        ("frame_index_time", 10.0, 10.0),
    ]
    assert [key for key, _ in expected] == expected_keys
    assert [key for key, _ in actual] == expected_keys
    assert [group["track_id"].tolist() for _, group in actual] == [
        [101],
        [102],
        [110],
    ]


def test_factor_graph_reassociation_accepts_numeric_string_timestamps() -> None:
    radar = pd.DataFrame(
        {
            "time_s": ["0.0", "1.0", "2.0"],
            "frame_index": ["0", "1", "2"],
            "track_id": [101, 102, 103],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [5.0, 5.0, 5.0],
        }
    )

    trajectory, selected = coordinate_descent_association_and_smoothing(
        radar,
        iterations=1,
        candidate_gate_m=1.0,
    )

    assert trajectory["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert selected["track_id"].tolist() == [101, 102, 103]
    assert selected["time_s"].tolist() == ["0.0", "1.0", "2.0"]
