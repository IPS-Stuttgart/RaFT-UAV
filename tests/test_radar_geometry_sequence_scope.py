from __future__ import annotations

import pandas as pd

from raft_uav.diagnostics.radar_geometry import (
    summarize_radar_geometry_audit,
    summarize_radar_geometry_by_track,
)


def _pooled_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["flight-a", "flight-a", "flight-b", "flight-b"],
            "track_id": [7, 7, 7, 7],
            "frame_index": [0, 1, 0, 1],
            "time_s": [0.0, 1.0, 10.0, 11.0],
            "geometry_delta_3d_m": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_geometry_summary_counts_reused_ids_per_sequence() -> None:
    summary = summarize_radar_geometry_audit(_pooled_audit())

    assert summary["rows"] == 4
    assert summary["track_ids"] == 2
    assert summary["frames"] == 4


def test_geometry_by_track_keeps_reused_track_ids_sequence_local() -> None:
    by_track = summarize_radar_geometry_by_track(_pooled_audit())
    by_track = by_track.sort_values("sequence_id").reset_index(drop=True)

    assert by_track["sequence_id"].tolist() == ["flight-a", "flight-b"]
    assert by_track["track_id"].tolist() == [7, 7]
    assert by_track["rows"].tolist() == [2, 2]
    assert by_track["time_s_min"].tolist() == [0.0, 10.0]
    assert by_track["time_s_max"].tolist() == [1.0, 11.0]


def test_geometry_summary_preserves_legacy_scope_without_sequence_ids() -> None:
    audit = _pooled_audit().drop(columns=["sequence_id"])

    summary = summarize_radar_geometry_audit(audit)
    by_track = summarize_radar_geometry_by_track(audit)

    assert summary["track_ids"] == 1
    assert summary["frames"] == 2
    assert by_track["rows"].tolist() == [4]
