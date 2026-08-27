from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.evaluation.oracle_coverage as oracle_coverage
from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


class _Node:
    def __init__(self, row: pd.Series) -> None:
        self.row = row


def test_oracle_coverage_selects_nearest_large_representable_candidate(
    monkeypatch,
) -> None:
    candidates = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "track_id": [1, 2],
            "track_index": [1, 2],
            "east_m": [9.0e307, 6.0e307],
            "north_m": [9.0e307, 8.0e307],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.9, 0.9],
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
        lambda **kwargs: [_Node(row) for _, row in kwargs["candidates"].iterrows()],
    )

    with np.errstate(over="raise", invalid="raise"):
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
    assert int(row["oracle_track_id"]) == 2
    assert float(row["oracle_truth_error_m"]) == pytest.approx(1.0e308)
    assert float(row["oracle_truth_error_2d_m"]) == pytest.approx(1.0e308)
    assert float(row["oracle_range_m"]) == pytest.approx(1.0e308)


def test_oracle_candidate_range_preserves_ordinary_norm_exactly() -> None:
    row = pd.Series({"east_m": 0.1, "north_m": 0.2, "up_m": 0.3})
    expected = float(np.linalg.norm(np.array([0.1, 0.2, 0.3])))

    assert oracle_coverage._candidate_range_m(row) == expected


def test_nearest_truth_time_delta_ignores_far_bracket_overflow() -> None:
    truth = pd.DataFrame({"time_s": [-1.0e308, 1.0e308]})

    with np.errstate(over="raise", invalid="raise"):
        delta = oracle_coverage._nearest_truth_time_delta_s(truth, 9.0e307)

    assert delta == pytest.approx(1.0e307)
