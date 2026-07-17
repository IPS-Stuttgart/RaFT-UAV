from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _AnchorState,
    _select_tracklet_viterbi_path,
)


def _radar_frame(frame_index: int, east_m: float) -> dict[str, object]:
    candidates = pd.DataFrame(
        [
            {
                "frame_index": frame_index,
                "time_s": float(frame_index),
                "track_id": 1,
                "east_m": east_m,
                "north_m": 0.0,
                "up_m": 0.0,
                "cat_prob_uav": 0.9,
            }
        ]
    )
    return {
        "kind": "radar",
        "time_s": float(frame_index),
        "candidates": candidates,
    }


def _anchor(east_m: float) -> _AnchorState:
    return _AnchorState(
        state=np.array([east_m, 0.0, 0.0, 10.0, 0.0, 0.0]),
        covariance=np.diag([5.0**2] * 6),
    )


def test_leading_gaps_cannot_collect_reacquisition_reward() -> None:
    events = [
        _radar_frame(0, 0.0),
        _radar_frame(1, 800.0),
        _radar_frame(2, 20.0),
    ]
    anchors = {0: _anchor(0.0), 1: _anchor(10.0), 2: _anchor(20.0)}
    config = TrackletViterbiAssociationConfig(
        missed_detection_cost=1.0,
        consecutive_miss_cost=1.0,
        anchor_nis_weight=2.0,
        track_switch_cost=10.0,
        reacquisition_miss_streak_threshold=2,
        reacquisition_reward=3.0,
        max_candidates_per_frame=4,
        range_gate_m=None,
    )

    selected = _select_tracklet_viterbi_path(
        events=events,
        anchors=anchors,
        covariance=np.diag([10.0**2] * 3),
        candidate_catprob_threshold=None,
        config=config,
    )

    assert [int(row["frame_index"]) for row in selected] == [0, 2]
    assert not bool(selected[0]["association_reacquisition_active"])
