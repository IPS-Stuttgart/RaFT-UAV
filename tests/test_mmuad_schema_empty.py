from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.schema import CANONICAL_TRUTH_COLUMNS, TruthFrame, normalize_truth_columns


def test_normalize_truth_columns_accepts_empty_table_without_columns() -> None:
    normalized = normalize_truth_columns(pd.DataFrame())

    assert normalized.empty
    assert list(normalized.columns) == list(CANONICAL_TRUTH_COLUMNS)
    TruthFrame(normalized).validate()
