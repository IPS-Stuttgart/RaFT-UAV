from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import CandidatePullConfig
from raft_uav.mmuad.candidate_pull import align_candidate_centers
from raft_uav.mmuad.candidate_pull import candidate_centers_for_results
from raft_uav.mmuad.candidate_pull import refine_official_results_with_candidate_pull


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA"],
            "Timestamp": [0.0, 100.0],
            "x_m": [1.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.5, 1.0],
        }
    )


@pytest.mark.parametrize(
    "time_tolerance_s",
    [np.nan, np.inf, -np.inf, -0.1, True, 1 + 0j, np.array([0.5]), np.ma.masked],
)
def test_candidate_centers_reject_invalid_time_tolerance(
    time_tolerance_s: object,
) -> None:
    with pytest.raises(ValueError, match="finite non-negative real scalar"):
        candidate_centers_for_results(
            _candidates(),
            _results()[["Sequence", "Timestamp"]],
            np.zeros((1, 3)),
            top_k=1,
            time_tolerance_s=time_tolerance_s,
        )


def test_alignment_rejects_invalid_time_tolerance() -> None:
    with pytest.raises(ValueError, match="finite non-negative real scalar"):
        align_candidate_centers(
            _results()[["Sequence", "Timestamp"]],
            pd.DataFrame(),
            time_tolerance_s=np.inf,
        )


def test_candidate_pull_rejects_unbounded_stale_frame_matching() -> None:
    with pytest.raises(ValueError, match="finite non-negative real scalar"):
        refine_official_results_with_candidate_pull(
            _results(),
            _candidates(),
            config=CandidatePullConfig(
                policy="constant",
                smoother="none",
                top_k=1,
                time_tolerance_s=np.inf,
            ),
        )


def test_zero_dimensional_zero_tolerance_keeps_exact_frame_only() -> None:
    centers = candidate_centers_for_results(
        _candidates(),
        _results()[["Sequence", "Timestamp"]],
        np.zeros((1, 3)),
        top_k=1,
        time_tolerance_s=np.array(0.0),
    )

    assert centers["candidate_count"].tolist() == [1]
    assert centers["top1_x"].tolist() == pytest.approx([1.0])


def test_exact_candidate_frame_ignores_overflowing_far_timestamp() -> None:
    candidates = pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA"],
            "Timestamp": [-1.0e308, 1.0e308],
            "x_m": [-10.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [1.0, 1.0],
        }
    )
    results = pd.DataFrame({"Sequence": ["seqA"], "Timestamp": [1.0e308]})

    with np.errstate(over="raise", invalid="raise"):
        centers = candidate_centers_for_results(
            candidates,
            results,
            np.zeros((1, 3), dtype=float),
            top_k=1,
            time_tolerance_s=0.0,
        )

    assert centers["candidate_count"].tolist() == [1]
    assert centers["top1_x"].tolist() == pytest.approx([10.0])
