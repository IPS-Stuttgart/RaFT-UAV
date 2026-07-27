from __future__ import annotations

from io import StringIO

import pandas as pd
import pytest

from raft_uav.mmuad import estimate_csv


@pytest.fixture(autouse=True)
def _force_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(estimate_csv, "_called_from_track5_estimate_grid", lambda: True)
    monkeypatch.setattr(estimate_csv, "_called_from_candidate_reservoir_cli", lambda: False)


def test_guard_preserves_headerless_integer_columns() -> None:
    rows = estimate_csv._read_csv_with_track5_estimate_grid_guard(
        StringIO("001,1\n002,2\n"),
        header=None,
    )

    assert rows.columns.tolist() == [0, 1]
    assert rows.iloc[:, 0].tolist() == ["001", "002"]


def test_guard_preserves_multiindex_columns() -> None:
    rows = estimate_csv._read_csv_with_track5_estimate_grid_guard(
        StringIO("identity,state\nsequence_id,x_m\n001,1\n"),
        header=[0, 1],
    )

    assert isinstance(rows.columns, pd.MultiIndex)
    assert rows.columns.tolist() == [("identity", "sequence_id"), ("state", "x_m")]
    assert rows.iloc[0, 0] == "001"
