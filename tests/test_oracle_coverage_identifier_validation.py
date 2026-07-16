from __future__ import annotations

import numpy as np
import pandas as pd

import raft_uav.evaluation.oracle_coverage as oracle_coverage
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


def test_optional_int_rejects_fractional_identifiers() -> None:
    assert oracle_coverage._optional_int(7.25) is None
    assert oracle_coverage._optional_int("8.5") is None
    assert oracle_coverage._optional_int(np.float64(-1.1)) is None


def test_optional_int_preserves_integer_equivalent_identifiers() -> None:
    assert oracle_coverage._optional_int(7) == 7
    assert oracle_coverage._optional_int(7.0) == 7
    assert oracle_coverage._optional_int("8.0") == 8


def test_oracle_coverage_does_not_truncate_fractional_candidate_ids(monkeypatch) -> None:
    candidates = pd.DataFrame(
        {
            "time_s": [0.0],
            "track_id": [12.75],
            "track_index": [3.5],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "cat_prob_uav": [0.9],
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

    monkeypatch.setattr(
        oracle_coverage._IMPL,
        "_nodes_for_radar_frame",
        lambda **kwargs: [_Node(kwargs["candidates"].iloc[0])],
    )

    row, retained = oracle_coverage._oracle_coverage_row(
        event_index=0,
        event={"time_s": 0.0},
        candidates=candidates,
        truth=pd.DataFrame(),
        anchor=None,
        covariance=np.eye(3),
        candidate_catprob_threshold=0.5,
        config=TrackletViterbiAssociationConfig(),
        truth_time_gate_s=1.0,
        previous_miss_streak=0,
    )

    assert retained
    assert pd.isna(row["oracle_track_id"])
    assert pd.isna(row["oracle_track_index"])
