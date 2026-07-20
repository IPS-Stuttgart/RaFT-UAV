from __future__ import annotations

import numpy as np
import pandas as pd

import raft_uav.evaluation.oracle_coverage as oracle_coverage
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


def test_oracle_coverage_preserves_large_identifier_keys(monkeypatch) -> None:
    oracle_track_id = 2**53 + 1
    retained_track_id = 2**53
    candidates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "track_id": [oracle_track_id, retained_track_id],
            "east_m": [0.0, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.1, 0.9],
        }
    )

    monkeypatch.setattr(
        oracle_coverage._IMPL,
        "_interpolated_truth_position",
        lambda *_args, **_kwargs: (np.zeros(3), 0.0, 0.0),
    )
    monkeypatch.setattr(
        oracle_coverage._IMPL,
        "_catprob_candidate_pool",
        lambda frame, _threshold: frame,
    )

    class _Node:
        def __init__(self, row: pd.Series) -> None:
            self.row = row

    def _nodes_for_radar_frame(**kwargs):
        rows = kwargs["candidates"]
        if kwargs["config"].max_candidates_per_frame == 1:
            return [_Node(rows.iloc[1])]
        return [_Node(rows.iloc[1]), _Node(rows.iloc[0])]

    monkeypatch.setattr(
        oracle_coverage._IMPL,
        "_nodes_for_radar_frame",
        _nodes_for_radar_frame,
    )

    row, retained = oracle_coverage._oracle_coverage_row(
        event_index=0,
        event={"time_s": 0.0},
        candidates=candidates,
        truth=pd.DataFrame(),
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=None,
        config=TrackletViterbiAssociationConfig(max_candidates_per_frame=1),
        truth_time_gate_s=1.0,
        previous_miss_streak=0,
    )

    assert not retained
    assert not bool(row["oracle_retained"])
    assert row["oracle_drop_reason"] == "top_k"
    assert row["oracle_track_id"] == oracle_track_id


def test_oracle_coverage_event_key_preserves_large_frame_index() -> None:
    candidates = pd.DataFrame({"frame_index": [str(2**53 + 1)]})

    assert oracle_coverage._event_key(candidates, 0.0) == (
        f"frame_index:{2**53 + 1}"
    )
