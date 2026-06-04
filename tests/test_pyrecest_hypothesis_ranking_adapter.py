from __future__ import annotations

from raft_uav.baselines.pyrecest_hypothesis_ranking import (
    GlobalTrackletHypothesisRankingConfig,
    rank_global_tracklet_replays,
)


def test_pyrecest_hypothesis_ranking_selects_replay_consistent_path() -> None:
    rows = rank_global_tracklet_replays(
        [
            {
                "path_id": "graph-rank-1",
                "graph_cost": 0.0,
                "selected_radar_rows": 100,
                "records": [{"nis": 200.0, "residual_norm_m": 1000.0}],
            },
            {
                "path_id": "graph-rank-2",
                "graph_cost": 10.0,
                "selected_radar_rows": 100,
                "records": [{"nis": 1.0, "residual_norm_m": 10.0}],
            },
        ],
        config=GlobalTrackletHypothesisRankingConfig(
            graph_cost_weight=0.1,
            replay_nis_weight=1.0,
            residual_weight=0.1,
        ),
    )

    assert rows[0]["path_id"] == "graph-rank-2"
    assert rows[0]["combined_objective"] == rows[0]["total_score"]


def test_pyrecest_hypothesis_ranking_maps_global_tracklet_summary_fields() -> None:
    rows = rank_global_tracklet_replays(
        [
            {
                "path_id": 7,
                "graph_cost": 1.0,
                "track_switches": 2,
                "missed_radar_count": 3,
                "unsupported_rf_count": 4,
                "hard_quarantined_segments_used": 1,
                "tail_duration_s": 5.0,
                "selected_radar_rows": 6,
                "nis_values": [1.0, 2.0],
                "residual_values": [10.0, 20.0],
                "selected_radar_track_ids": "2163,2272",
            }
        ]
    )

    row = rows[0]
    assert row["path_id"] == 7
    assert row["track_switches"] == 2
    assert row["missed_detection_count"] == 3
    assert row["unsupported_measurement_count"] == 4
    assert row["hard_quarantine_count"] == 1
    assert row["coverage_count"] == 6
    assert row["metadata_selected_radar_track_ids"] == "2163,2272"
