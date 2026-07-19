from __future__ import annotations

import pandas as pd

from raft_uav.baselines.topk_weakz_tracklet import (
    TopKWeakZTrackletConfig,
    build_fortem_tracklets,
)


def _radar_rows(track_ids: list[object]) -> pd.DataFrame:
    rows = []
    for track_position, track_id in enumerate(track_ids):
        for time_s in range(3):
            rows.append(
                {
                    "track_id": track_id,
                    "time_s": float(time_s),
                    "east_m": float(100 * track_position + time_s),
                    "north_m": 0.0,
                    "up_m": 100.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_build_fortem_tracklets_keeps_adjacent_large_integer_ids_distinct() -> None:
    track_ids = [2**53, 2**53 + 1]

    tracklets = build_fortem_tracklets(
        _radar_rows(track_ids),
        TopKWeakZTrackletConfig(min_tracklet_length=3),
    )

    assert len(tracklets) == 2
    assert {tracklet.track_id for tracklet in tracklets} == set(track_ids)
    assert sorted(tracklet.row_count for tracklet in tracklets) == [3, 3]


def test_build_fortem_tracklets_does_not_stitch_fractional_track_ids() -> None:
    tracklets = build_fortem_tracklets(
        _radar_rows([7.5]),
        TopKWeakZTrackletConfig(min_tracklet_length=3),
    )

    assert tracklets == []
