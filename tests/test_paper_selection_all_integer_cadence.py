from __future__ import annotations

import pandas as pd

from raft_uav.paper_selection import select_paper_compatible_radar_track


def test_all_integer_timestamps_use_observed_acquisition_cadence() -> None:
    radar = pd.DataFrame(
        [
            *_rows(track_id=1, times=[0.0, 2.0, 4.0]),
            *_rows(track_id=2, times=[0.0, 4.0]),
        ]
    )

    selected = select_paper_compatible_radar_track(
        radar,
        range_gate_m=None,
        catprob_threshold=None,
    )

    assert selected["track_id"].astype(int).unique().tolist() == [1]
    assert selected["time_s"].astype(float).tolist() == [0.0, 2.0, 4.0]


def _rows(*, track_id: int, times: list[float]) -> list[dict[str, float | int]]:
    return [
        {
            "time_s": float(time_s),
            "track_id": int(track_id),
            "track_index": 0,
            "east_m": float(track_id * 10 + row_index),
            "north_m": float(row_index),
            "up_m": 10.0,
        }
        for row_index, time_s in enumerate(times)
    ]
