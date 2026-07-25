from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import load_submission_csv


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB"],
            "time_s": [0.0, 1.0, 0.0],
            "track_id": ["uav0", "uav0", "uav1"],
            "x_m": [10.0, 11.0, 20.0],
            "y_m": [30.0, 31.0, 40.0],
            "z_m": [50.0, 51.0, 60.0],
            "score": [0.9, 0.8, 0.7],
        }
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("time_s", "not-a-time"),
        ("x_m", np.inf),
        ("y_m", -np.inf),
        ("z_m", "not-a-number"),
    ],
)
def test_load_submission_csv_rejects_malformed_required_numeric_rows(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    path = tmp_path / "submission.csv"
    rows = _submission_rows()
    rows[column] = rows[column].astype(object)
    rows.loc[1, column] = value
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match=rf"invalid submission {column} row") as exc_info:
        load_submission_csv(path)

    assert "index/indices 1" in str(exc_info.value)


def test_load_submission_csv_preserves_every_valid_row(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    _submission_rows().to_csv(path, index=False)

    loaded = load_submission_csv(path)

    assert len(loaded) == 3
    assert loaded[["sequence_id", "time_s"]].to_records(index=False).tolist() == [
        ("seqA", 0.0),
        ("seqA", 1.0),
        ("seqB", 0.0),
    ]
