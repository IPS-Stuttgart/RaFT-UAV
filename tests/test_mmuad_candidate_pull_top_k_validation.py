from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import CandidatePullConfig
from raft_uav.mmuad.candidate_pull import refine_official_results_with_candidate_pull
from raft_uav.mmuad.candidate_pull import topk_candidate_centers


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 0.0],
            "x_m": [1.0, 2.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.9, 0.8],
        }
    )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, np.nan, np.inf])
def test_topk_candidate_centers_rejects_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        topk_candidate_centers(_candidates(), top_k=top_k)


def test_candidate_pull_orchestration_rejects_fractional_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        refine_official_results_with_candidate_pull(
            _results(),
            _candidates(),
            config=CandidatePullConfig(policy="none", smoother="none", top_k=1.5),
        )


def test_integer_equivalent_top_k_preserves_existing_selection() -> None:
    centers = topk_candidate_centers(_candidates(), top_k=2.0)

    assert centers["topk_candidate_count"].tolist() == [2]
