from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.time_offset import radar_frame_groups


def test_radar_frame_groups_sorts_serialized_frame_indices_numerically() -> None:
    large_frame_id = 2**80 + 123
    radar = pd.DataFrame(
        {
            "frame_index": ["10", str(large_frame_id), "2"],
            "track_id": [1, 2, 3],
        }
    )

    groups = radar_frame_groups(radar)

    assert [int(group["frame_index"].iloc[0]) for group in groups] == [
        2,
        10,
        large_frame_id,
    ]


def test_radar_frame_groups_sorts_serialized_times_numerically() -> None:
    radar = pd.DataFrame(
        {
            "time_s": ["10", "2"],
            "track_id": [1, 2],
        }
    )

    groups = radar_frame_groups(radar)

    assert [float(group["time_s"].iloc[0]) for group in groups] == [2.0, 10.0]


def test_radar_frame_groups_requires_a_physical_frame_key() -> None:
    radar = pd.DataFrame({"track_id": [1, 2]})

    with pytest.raises(KeyError, match="time_s or frame_index"):
        radar_frame_groups(radar)
