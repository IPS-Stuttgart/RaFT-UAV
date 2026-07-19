import numpy as np
import pandas as pd

from raft_uav.diagnostics.time_offset import radar_frame_groups


def test_radar_frame_groups_preserves_indexed_frames_with_partial_metadata():
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "frame_index": [10, 11, np.nan, np.nan],
            "track_id": [1, 2, 1, 2],
        }
    )

    groups = radar_frame_groups(radar)

    assert [len(group) for group in groups] == [1, 1, 2]
    assert [int(groups[index]["frame_index"].iloc[0]) for index in (0, 1)] == [
        10,
        11,
    ]
    assert groups[2]["frame_index"].isna().all()
    assert [group["time_s"].iloc[0] for group in groups] == [0.0, 0.0, 1.0]
    assert sum(len(group) for group in groups) == len(radar)
