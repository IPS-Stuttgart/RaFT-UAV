from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_retention import (
    _catprob_threshold_penalty,
    _nodes_for_radar_frame_with_track_retention,
    _track_support_by_id,
    _track_support_cost,
)


def _radar_frame(frame_index: int, rows: list[dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["frame_index"] = frame_index
    frame["time_s"] = float(frame_index)
    frame["up_m"] = frame.get("up_m", 0.0)
    return frame


def test_track_aware_retention_keeps_per_track_representatives() -> None:
    candidates = _radar_frame(
        0,
        [
            {"track_id": 10, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99},
            {"track_id": 11, "east_m": 1.0, "north_m": 0.0, "cat_prob_uav": 0.95},
            {"track_id": 12, "east_m": 2.0, "north_m": 0.0, "cat_prob_uav": 0.05},
        ],
    )
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=0,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=config,
    )

    retained_track_ids = {node.track_id for node in nodes if not node.is_miss}
    assert retained_track_ids == {10, 11, 12}


def test_track_aware_retention_keeps_below_threshold_track_with_soft_penalty() -> None:
    candidates = _radar_frame(
        0,
        [
            {"track_id": 10, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99},
            {"track_id": 11, "east_m": 1.0, "north_m": 0.0, "cat_prob_uav": 0.95},
            {"track_id": 12, "east_m": 2.0, "north_m": 0.0, "cat_prob_uav": 0.05},
        ],
    )
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=0,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=0.4,
        config=config,
    )

    retained = {node.track_id: node for node in nodes if not node.is_miss}
    assert set(retained) == {10, 11, 12}
    low_catprob = retained[12]
    assert low_catprob.row is not None
    assert bool(low_catprob.row["association_catprob_below_threshold"])
    assert float(low_catprob.row["association_catprob_soft_penalty"]) > 0.0


def test_catprob_threshold_penalty_increases_with_gap() -> None:
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    near = pd.Series({"cat_prob_uav": 0.35})
    far = pd.Series({"cat_prob_uav": 0.05})

    near_penalty = _catprob_threshold_penalty(
        near,
        candidate_catprob_threshold=0.4,
        config=config,
    )
    far_penalty = _catprob_threshold_penalty(
        far,
        candidate_catprob_threshold=0.4,
        config=config,
    )

    assert near_penalty > 0.0
    assert far_penalty > near_penalty


def test_track_support_prior_rewards_longer_continuous_tracks() -> None:
    radar = pd.DataFrame(
        [
            {"track_id": 1, "frame_index": 0, "time_s": 0.0, "cat_prob_uav": 0.99},
            {"track_id": 2, "frame_index": 0, "time_s": 0.0, "cat_prob_uav": 0.20},
            {"track_id": 2, "frame_index": 1, "time_s": 1.0, "cat_prob_uav": 0.20},
            {"track_id": 2, "frame_index": 2, "time_s": 2.0, "cat_prob_uav": 0.20},
            {"track_id": 2, "frame_index": 3, "time_s": 3.0, "cat_prob_uav": 0.20},
        ]
    )
    support = _track_support_by_id(radar)
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    short_cost, _ = _track_support_cost(
        pd.Series({"track_id": 1}),
        track_support_by_id=support,
        config=config,
    )
    long_cost, long_support = _track_support_cost(
        pd.Series({"track_id": 2}),
        track_support_by_id=support,
        config=config,
    )

    assert long_support["count"] == 4.0
    assert long_support["continuity"] == 1.0
    assert long_cost < short_cost


def test_track_support_diagnostics_are_added_to_retained_rows() -> None:
    candidates = _radar_frame(
        3,
        [
            {"track_id": 1, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99},
            {"track_id": 2, "east_m": 1.0, "north_m": 0.0, "cat_prob_uav": 0.20},
        ],
    )
    support = {
        2: {
            "count": 4.0,
            "span_s": 3.0,
            "frame_span": 4.0,
            "continuity": 1.0,
            "median_catprob": 0.2,
            "score": 3.0,
        }
    }
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=3,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=0.4,
        config=config,
        track_support_by_id=support,
    )

    retained = {node.track_id: node for node in nodes if not node.is_miss}
    supported = retained[2]
    assert supported.row is not None
    assert float(supported.row["association_track_support_cost"]) < 0.0
    assert float(supported.row["association_track_support_count"]) == 4.0
    assert float(supported.row["association_track_support_continuity"]) == 1.0


def test_track_aware_retention_still_keeps_missed_detection_node() -> None:
    candidates = _radar_frame(
        0,
        [{"track_id": 1, "east_m": 0.0, "north_m": 0.0, "cat_prob_uav": 0.99}],
    )
    config = TrackletViterbiAssociationConfig(max_candidates_per_frame=1, range_gate_m=None)

    nodes = _nodes_for_radar_frame_with_track_retention(
        event_index=0,
        candidates=candidates,
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=config,
    )

    assert any(node.is_miss for node in nodes)
