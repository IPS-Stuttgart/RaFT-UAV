import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.radar_fingerprint import _continuous_track_segments


@pytest.mark.parametrize("invalid_frame_index", [np.nan, np.inf])
def test_partial_frame_indices_fall_back_to_timestamp_continuity(invalid_frame_index):
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0.0, invalid_frame_index, 2.0],
            "time_s": [0.0, 1.0, 10.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert [segment["time_s"].tolist() for segment in segments] == [
        [0.0, 1.0],
        [10.0],
    ]
    assert all(
        float(segment["time_s"].iloc[-1]) >= float(segment["time_s"].iloc[0])
        for segment in segments
    )


def test_complete_frame_indices_remain_authoritative_for_continuity():
    radar = pd.DataFrame(
        {
            "track_id": [7, 7, 7],
            "frame_index": [0.0, 1.0, 2.0],
            "time_s": [0.0, 100.0, 200.0],
        }
    )

    segments = _continuous_track_segments(radar)

    assert len(segments) == 1
    assert segments[0]["time_s"].tolist() == [0.0, 100.0, 200.0]
