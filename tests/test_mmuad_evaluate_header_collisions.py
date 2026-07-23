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


def test_load_submission_csv_checks_headers_before_time_normalization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        [[0.0, 1.0, "seqA", 1.0, 2.0, 3.0]],
        columns=[" Time_S ", "time_s", "sequence_id", "x_m", "y_m", "z_m"],
    ).to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="ambiguous columns after trimming whitespace and ignoring case",
    ):
        load_submission_csv(path)
