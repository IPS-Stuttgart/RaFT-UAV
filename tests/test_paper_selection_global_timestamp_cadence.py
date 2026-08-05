from __future__ import annotations

import pandas as pd

from raft_uav.paper_selection import select_paper_compatible_radar_track


def _row(*, track_id: int, time_s: float) -> dict[str, float | int]:
    return {
        "time_s": time_s,
        "track_id": track_id,
        "track_index": 0,
        "east_m": float(track_id * 10),
        "north_m": time_s,
        "up_m": 10.0,
        "range_m": 500.0,
        "cat_prob_uav": 0.8,
    }


def test_timestamp_only_track_continuity_uses_acquisition_cadence() -> None:
    radar = pd.DataFrame(
        [
            *[
                _row(track_id=8, time_s=time_s)
                for time_s in (0.25, 10.25, 20.25, 30.25, 40.25, 50.25, 60.25)
            ],
            *[
                _row(track_id=9, time_s=time_s)
                for time_s in (1.25, 2.25, 3.25, 4.25, 5.25)
            ],
        ]
    )

    selected = select_paper_compatible_radar_track(
        radar,
        range_gate_m=800.0,
        catprob_threshold=None,
    )

    assert selected["track_id"].astype(int).unique().tolist() == [9]
    assert selected["time_s"].tolist() == [1.25, 2.25, 3.25, 4.25, 5.25]
    assert selected["association_segment_frames"].max() == 5
