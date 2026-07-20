from __future__ import annotations

import numpy as np
import pandas as pd

import raft_uav.evaluation.oracle_coverage as oracle_coverage
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


class _Node:
    def __init__(self, row: pd.Series) -> None:
        self.row = row


def test_oracle_coverage_preserves_large_candidate_identifier_identity(
    monkeypatch,
) -> None:
    frame_index = 2**80 + 17
    oracle_track_id = 2**80 + 101
    retained_track_id = oracle_track_id + 1
    candidates = pd.DataFrame(
        {
            "frame_index": [str(frame_index), str(frame_index)],
            "track_id": [str(oracle_track_id), str(retained_track_id)],
            "time_s": [0.0, 0.0],
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
    monkeypatch.setattr(
        oracle_coverage._IMPL,
        "_nodes_for_radar_frame",
        lambda **kwargs: [_Node(kwargs["candidates"].iloc[1])],
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
    assert row["event_key"] == f"frame_index:{frame_index}"
