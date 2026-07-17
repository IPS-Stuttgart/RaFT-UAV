from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.learned_tracklet_viterbi import (
    _select_learned_tracklet_viterbi_path,
)
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _ViterbiNode,
)


def _candidate(frame_index: int, *, unary_cost: float = 0.25) -> _ViterbiNode:
    row = pd.Series(
        {
            "frame_index": frame_index,
            "track_id": 1,
            "time_s": float(frame_index),
            "east_m": float(frame_index * 10),
            "north_m": 0.0,
            "up_m": 0.0,
        }
    )
    return _ViterbiNode(
        event_index=frame_index,
        event_key=("frame_index", frame_index),
        time_s=float(frame_index),
        row=row,
        position=np.array([float(frame_index * 10), 0.0, 0.0]),
        velocity=None,
        track_id=1,
        unary_cost=unary_cost,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
        has_anchor=True,
    )


def _miss(frame_index: int) -> _ViterbiNode:
    return _ViterbiNode(
        event_index=frame_index,
        event_key=("frame_index", frame_index),
        time_s=float(frame_index),
        row=None,
        position=None,
        velocity=None,
        track_id=None,
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
        is_miss=True,
    )


def test_learned_tracklet_viterbi_does_not_reward_leading_misses(
    monkeypatch,
) -> None:
    frames = [
        [_candidate(0), _miss(0)],
        [_miss(1)],
        [_candidate(2), _miss(2)],
    ]

    def nodes_for_frame(*, event_index: int, **_kwargs) -> list[_ViterbiNode]:
        return frames[event_index]

    monkeypatch.setattr(
        "raft_uav.baselines.learned_tracklet_viterbi._learned_nodes_for_radar_frame",
        nodes_for_frame,
    )
    config = TrackletViterbiAssociationConfig(
        missed_detection_cost=1.0,
        consecutive_miss_cost=0.0,
        reacquisition_miss_streak_threshold=2,
        reacquisition_reward=3.0,
        range_gate_m=None,
    )
    events = [
        {"kind": "radar", "candidates": pd.DataFrame()}
        for _frame_index in range(3)
    ]

    selected = _select_learned_tracklet_viterbi_path(
        events=events,
        anchors={},
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=config,
        model=object(),
        learned_unary_weight=1.0,
        hand_unary_weight=1.0,
    )

    assert [int(row["frame_index"]) for row in selected] == [0, 2]
