from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import candidate_centers_for_results
from raft_uav.mmuad.candidate_pull import topk_candidate_centers


def _large_score_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 0.0],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [1.0e308, 1.0e308],
            "confidence": [0.9, 0.8],
        }
    )


def test_topk_candidate_centers_normalizes_large_finite_scores() -> None:
    centers = topk_candidate_centers(_large_score_candidates(), top_k=2)

    row = centers.iloc[0]
    assert row["weighted5_x"] == pytest.approx(5.0)
    assert row["weighted5_y"] == pytest.approx(0.0)
    assert row["weighted5_z"] == pytest.approx(0.0)
    assert row["top_score"] == pytest.approx(1.0e308)
    assert row["top_score_margin"] == pytest.approx(0.0)
    assert np.isfinite(
        row[["weighted5_x", "weighted5_y", "weighted5_z"]].to_numpy(dtype=float)
    ).all()


def test_rowwise_candidate_centers_normalizes_large_finite_scores() -> None:
    results = pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})
    centers = candidate_centers_for_results(
        _large_score_candidates(),
        results,
        np.zeros((1, 3), dtype=float),
        top_k=2,
        time_tolerance_s=0.1,
    )

    row = centers.iloc[0]
    assert row["weighted5_x"] == pytest.approx(5.0)
    assert row["weighted10_x"] == pytest.approx(5.0)
    assert row["top_score"] == pytest.approx(1.0e308)
    assert row["top_score_margin"] == pytest.approx(0.0)
    assert np.isfinite(
        row[
            [
                "weighted5_x",
                "weighted5_y",
                "weighted5_z",
                "weighted10_x",
                "weighted10_y",
                "weighted10_z",
            ]
        ].to_numpy(dtype=float)
    ).all()
