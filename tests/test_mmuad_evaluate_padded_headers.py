from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import load_submission_csv


def test_load_submission_csv_accepts_padded_alias_headers(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            " Sequence ": ["seqA"],
            " Timestamp_S ": ["1.25"],
            " Track ": ["uav0"],
            " X ": ["10.0"],
            " Y ": ["20.0"],
            " Z ": ["30.0"],
            " Confidence ": ["0.75"],
        }
    ).to_csv(path, index=False)

    rows = load_submission_csv(path)

    assert rows.loc[0, "sequence_id"] == "seqA"
    assert rows.loc[0, "track_id"] == "uav0"
    assert float(rows.loc[0, "time_s"]) == pytest.approx(1.25)
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [10.0, 20.0, 30.0]
    assert float(rows.loc[0, "score"]) == pytest.approx(0.75)
