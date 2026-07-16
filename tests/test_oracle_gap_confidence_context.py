from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import confidence_diagnostics


def test_confidence_context_preserves_estimate_schema_order_and_index() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": ["1.0", "0.0"],
            "track_id": [101, 100],
            "association_score": [0.2, 0.1],
            "nis": [1.0, 2.0],
        },
        index=pd.Index([10, 4], name="estimate_row"),
    )
    selected = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "track_id": [200, 201],
            "association_score": [2.0, 3.0],
            "association_nis": [4.0, 5.0],
        }
    )

    diagnostics = confidence_diagnostics(estimates, selected)

    assert diagnostics.index.equals(estimates.index)
    assert diagnostics["time_s"].tolist() == ["1.0", "0.0"]
    assert diagnostics["track_id"].tolist() == [101, 100]
    assert diagnostics["association_score"].tolist() == [0.2, 0.1]
    assert diagnostics["selected_context_track_id"].tolist() == [201, 200]
    assert diagnostics["selected_context_association_score"].tolist() == [3.0, 2.0]
    assert diagnostics["selected_context_association_nis"].tolist() == [5.0, 4.0]
    assert "track_id_x" not in diagnostics.columns
    assert "track_id_y" not in diagnostics.columns


def test_confidence_context_keeps_rows_with_invalid_times() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, np.nan, "invalid"],
            "track_id": [100, 101, 102],
        }
    )
    selected = pd.DataFrame(
        {
            "time_s": [0.0, np.nan],
            "track_id": [200, 999],
        }
    )

    diagnostics = confidence_diagnostics(estimates, selected)

    assert diagnostics["track_id"].tolist() == [100, 101, 102]
    assert diagnostics.loc[0, "selected_context_track_id"] == 200
    assert diagnostics.loc[[1, 2], "selected_context_track_id"].isna().all()
