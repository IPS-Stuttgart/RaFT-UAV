from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.evaluate import load_submission_csv


def test_submission_csv_rejects_exact_duplicate_physical_headers(
    tmp_path: Path,
) -> None:
    submission_csv = tmp_path / "submission.csv"
    submission_csv.write_text(
        "sequence_id,time_s,x_m,x_m,y_m,z_m\n"
        "sequence-001,0.0,1.0,999.0,2.0,3.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"ambiguous columns after trimming whitespace and ignoring case: 'x_m'",
    ):
        load_submission_csv(submission_csv)
