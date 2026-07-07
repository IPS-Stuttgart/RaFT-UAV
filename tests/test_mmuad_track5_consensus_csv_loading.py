from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_consensus_ensemble import _read_estimate_csv


def test_consensus_estimate_csv_reader_preserves_text_identifiers(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            " Sequence ": ["001"],
            " Timestamp ": ["0.0"],
            " x ": ["1.0"],
        }
    ).to_csv(estimate_csv, index=False)

    rows = _read_estimate_csv(estimate_csv)

    assert rows.loc[0, "Sequence"] == "001"
    assert "Timestamp" in rows.columns
    assert "x" in rows.columns
