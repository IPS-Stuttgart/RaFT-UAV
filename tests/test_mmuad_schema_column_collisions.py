from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


def test_candidate_normalizer_rejects_case_and_whitespace_column_collision() -> None:
    raw = pd.DataFrame(
        {
            "sequence_id": ["seq0"],
            "time_s": [0.0],
            "source": ["radar"],
            "X_M": [1.0],
            " x_m ": [99.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"column names are ambiguous.*X_M.* x_m .*x_m",
    ):
        normalize_candidate_columns(raw)


def test_truth_normalizer_rejects_exact_duplicate_columns() -> None:
    raw = pd.DataFrame(
        [["seq0", 0.0, 1.0, 2.0, 3.0, 4.0]],
        columns=["sequence_id", "time_s", "time_s", "x_m", "y_m", "z_m"],
    )

    with pytest.raises(
        ValueError,
        match=r"column names are ambiguous.*time_s.*time_s",
    ):
        normalize_truth_columns(raw)
