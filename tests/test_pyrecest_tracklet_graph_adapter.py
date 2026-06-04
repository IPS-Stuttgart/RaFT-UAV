from __future__ import annotations

from raft_uav.baselines.pyrecest_tracklet_graph import (
    FortemTrackletGraphConfig,
    fortem_tracklet_from_rows,
    fortem_tracklet_from_summary,
    fortem_tracklet_paths_to_rows,
    rank_fortem_tracklet_paths,
)


def test_fortem_tracklet_from_summary_preserves_track_id_and_cost() -> None:
    row = {
        "segment_id": 7,
        "track_id": 2272,
        "start_time_s": 10.0,
        "end_time_s": 12.0,
        "start_east_m": 1.0,
        "start_north_m": 2.0,
        "start_up_m": 3.0,
        "end_east_m": 4.0,
        "end_north_m": 5.0,
        "end_up_m": 6.0,
        "score": 9.5,
    }

    tracklet = fortem_tracklet_from_summary(row)

    assert tracklet.id == 7
    assert tracklet.metadata["track_id"] == 2272
    assert tracklet.cost == 9.5


def test_rank_fortem_tracklet_paths_uses_pyrecest_graph_search() -> None:
    good_a = fortem_tracklet_from_rows(
        "a",
        [
            {"time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
            {"time_s": 1.0, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
        ],
    )
    good_b = fortem_tracklet_from_rows(
        "b",
        [
            {"time_s": 2.0, "east_m": 2.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
            {"time_s": 3.0, "east_m": 3.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
        ],
    )
    too_fast = fortem_tracklet_from_rows(
        "bad",
        [
            {"time_s": 2.0, "east_m": 1000.0, "north_m": 0.0, "up_m": 0.0, "track_id": 2},
            {"time_s": 3.0, "east_m": 1001.0, "north_m": 0.0, "up_m": 0.0, "track_id": 2},
        ],
    )

    paths = rank_fortem_tracklet_paths(
        [good_a, good_b, too_fast],
        config=FortemTrackletGraphConfig(
            top_k_paths=3,
            max_transition_speed_mps=50.0,
            speed_weight=0.0,
            coverage_reward_per_row=1.0,
        ),
    )

    assert paths[0].tracklet_ids == ("a", "b")
    assert ("a", "bad") not in [path.tracklet_ids for path in paths]


def test_path_rows_match_raft_uav_diagnostic_names() -> None:
    first = fortem_tracklet_from_rows(
        "a",
        [
            {"time_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
            {"time_s": 1.0, "east_m": 1.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
        ],
    )
    second = fortem_tracklet_from_rows(
        "b",
        [
            {"time_s": 2.0, "east_m": 2.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
            {"time_s": 3.0, "east_m": 3.0, "north_m": 0.0, "up_m": 0.0, "track_id": 1},
        ],
    )
    paths = rank_fortem_tracklet_paths(
        [first, second],
        config=FortemTrackletGraphConfig(speed_weight=0.0, coverage_reward_per_row=1.0),
    )

    rows = fortem_tracklet_paths_to_rows(paths, tracklets={"a": first, "b": second})

    assert rows[0]["path_id"] == 0
    assert rows[0]["segment_ids"] == "a;b"
    assert rows[0]["coverage_segments"] == 2
    assert "graph_cost" in rows[0]
