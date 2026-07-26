from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import (
    align_rowwise_candidate_centers,
    candidate_centers_for_results,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq1", "seq1", "seq1", "seq1"],
            "Timestamp": [0.0, 0.0, 1.0, 1.0],
            "x_m": [0.0, 100.0, 10.0, 110.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [1.0, 1.0, 1.0, 1.0],
            "confidence": [1.0, 1.0, 1.0, 1.0],
            "cluster_point_count": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_direct_candidate_centers_use_positions_not_integer_index_labels() -> None:
    results = pd.DataFrame(
        {"Sequence": ["seq1", "seq1"], "Timestamp": [0.0, 1.0]},
        index=[7, 42],
    )
    current_xyz = np.array([[1.0, 0.0, 0.0], [109.0, 0.0, 0.0]])

    centers = candidate_centers_for_results(
        _candidates(), results, current_xyz, top_k=2, time_tolerance_s=0.1
    )

    assert centers["row_index"].tolist() == [7, 42]
    assert centers["nearest_current_x"].tolist() == pytest.approx([0.0, 110.0])
    aligned = align_rowwise_candidate_centers(results, centers)
    assert aligned.loc[7, "nearest_current_x"] == pytest.approx(0.0)
    assert aligned.loc[42, "nearest_current_x"] == pytest.approx(110.0)


def test_direct_candidate_centers_preserve_text_index_labels() -> None:
    results = pd.DataFrame(
        {"Sequence": ["seq1", "seq1"], "Timestamp": [0.0, 1.0]},
        index=["first", "second"],
    )
    current_xyz = np.array([[1.0, 0.0, 0.0], [109.0, 0.0, 0.0]])

    centers = candidate_centers_for_results(
        _candidates(), results, current_xyz, top_k=2, time_tolerance_s=0.1
    )

    assert centers["row_index"].tolist() == ["first", "second"]


def test_direct_candidate_centers_reject_ambiguous_or_misaligned_rows() -> None:
    duplicate_index = pd.DataFrame(
        {"Sequence": ["seq1", "seq1"], "Timestamp": [0.0, 1.0]},
        index=[3, 3],
    )
    with pytest.raises(ValueError, match="results index must be unique"):
        candidate_centers_for_results(
            _candidates(), duplicate_index, np.zeros((2, 3))
        )

    results = duplicate_index.reset_index(drop=True)
    with pytest.raises(ValueError, match=r"current_xyz must have shape"):
        candidate_centers_for_results(_candidates(), results, np.zeros((1, 3)))
