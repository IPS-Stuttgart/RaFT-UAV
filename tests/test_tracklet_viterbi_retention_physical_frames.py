from __future__ import annotations

import math

import pandas as pd

from raft_uav.baselines.tracklet_viterbi_retention import _track_support_by_id


def test_duplicate_rows_do_not_inflate_track_support() -> None:
    radar = pd.DataFrame(
        [
            {
                "track_id": 5,
                "frame_index": 0,
                "time_s": 0.0,
                "cat_prob_uav": 0.1,
            },
            {
                "track_id": 5,
                "frame_index": 0,
                "time_s": 0.0,
                "cat_prob_uav": 0.9,
            },
            {
                "track_id": 5,
                "frame_index": 1,
                "time_s": 1.0,
                "cat_prob_uav": 0.5,
            },
            {
                "track_id": 6,
                "frame_index": 0,
                "time_s": 0.0,
                "cat_prob_uav": 0.5,
            },
            {
                "track_id": 6,
                "frame_index": 1,
                "time_s": 1.0,
                "cat_prob_uav": 0.5,
            },
        ]
    )

    support = _track_support_by_id(radar)

    assert support[5]["count"] == 2.0
    assert support[5]["continuity"] == 1.0
    assert support[5]["median_catprob"] == 0.5
    assert support[6]["count"] == 2.0
    assert math.isclose(support[5]["score"], support[6]["score"], rel_tol=1.0e-12)


def test_reused_frame_index_at_distinct_times_counts_distinct_frames() -> None:
    radar = pd.DataFrame(
        [
            {
                "track_id": 5,
                "frame_index": 0,
                "time_s": 0.0,
                "cat_prob_uav": 0.5,
            },
            {
                "track_id": 5,
                "frame_index": 0,
                "time_s": 1.0,
                "cat_prob_uav": 0.5,
            },
        ]
    )

    support = _track_support_by_id(radar)

    assert support[5]["count"] == 2.0
    assert support[5]["span_s"] == 1.0
