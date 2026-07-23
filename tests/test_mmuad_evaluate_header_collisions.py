from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import load_submission_csv


@pytest.mark.parametrize(
    ("first_header", "second_header"),
    [
        ("X", " x "),
        ("sequence_id", " SEQUENCE_ID "),
    ],
)
def test_load_submission_csv_rejects_case_whitespace_header_collisions(
    tmp_path: Path,
    first_header: str,
    second_header: str,
) -> None:
    path = tmp_path / "submission.csv"
    row = {
        "sequence_id": "seqA",
        "time_s": 0.0,
        "x_m": 1.0,
        "y_m": 2.0,
        "z_m": 3.0,
    }
    row[first_header] = "first"
    row[second_header] = "second"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="ambiguous columns after trimming whitespace and ignoring case",
    ) as exc_info:
        load_submission_csv(path)

    message = str(exc_info.value)
    assert first_header in message
    assert second_header in message
