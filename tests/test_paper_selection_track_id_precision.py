from __future__ import annotations

import pandas as pd

from raft_uav.paper_selection import (
    _track_id_from_frame,
    select_paper_compatible_radar_track,
)


def test_paper_preselector_preserves_large_serialized_track_id_metadata() -> None:
    track_id = 2**80 + 123
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "frame_index": [0, 1],
            "track_id": [str(track_id), str(track_id)],
            "track_index": [0, 0],
            "east_m": [10.0, 11.0],
            "north_m": [20.0, 21.0],
            "up_m": [30.0, 31.0],
        }
    )

    selected = select_paper_compatible_radar_track(
        radar,
        range_gate_m=None,
        catprob_threshold=None,
    )

    assert selected["association_segment_track_id"].tolist() == [track_id, track_id]
    assert selected["association_preselector_track_id"].tolist() == [track_id, track_id]


def test_paper_selection_does_not_truncate_fractional_track_ids() -> None:
    frame = pd.DataFrame({"track_id": ["7.5"]})

    assert _track_id_from_frame(frame) == -1
