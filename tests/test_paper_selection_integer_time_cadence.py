from __future__ import annotations

import pandas as pd

from raft_uav.paper_selection import select_paper_compatible_radar_track


def test_integer_timestamps_use_time_cadence_without_frame_indices() -> None:
    radar = pd.DataFrame(
        [
            *_rows(track_id=1, times=[0.0, 2.0, 4.0]),
            *_rows(track_id=2, times=[0.5, 1.0]),
        ]
    )

    selected = select_paper_compatible_radar_track(
        radar,
        range_gate_m=None,
        catprob_threshold=None,
    )

    assert selected["track_id"].astype(int).unique().tolist() == [1]
    assert selected["time_s"].astype(float).tolist() == [0.0, 2.0, 4.0]


def test_integer_frame_indices_keep_strict_gap_semantics() -> None:
    radar = pd.DataFrame(
        [
            *_rows(
                track_id=1,
                times=[0.0, 2.0, 4.0],
                frame_indices=[0, 2, 4],
            ),
            *_rows(
                track_id=2,
                times=[0.5, 1.0],
                frame_indices=[10, 11],
            ),
        ]
    )

    selected = select_paper_compatible_radar_track(
        radar,
        range_gate_m=None,
        catprob_threshold=None,
    )

    assert selected["track_id"].astype(int).unique().tolist() == [2]
    assert selected["frame_index"].astype(int).tolist() == [10, 11]


def _rows(
    *,
    track_id: int,
    times: list[float],
    frame_indices: list[int] | None = None,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for row_index, time_s in enumerate(times):
        row: dict[str, float | int] = {
            "time_s": float(time_s),
            "track_id": int(track_id),
            "track_index": 0,
            "east_m": float(track_id * 10 + row_index),
            "north_m": float(row_index),
            "up_m": 10.0,
        }
        if frame_indices is not None:
            row["frame_index"] = int(frame_indices[row_index])
        rows.append(row)
    return rows
