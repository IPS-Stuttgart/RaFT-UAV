from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.evaluate import load_submission_csv


@pytest.mark.parametrize(
    ("row", "canonical"),
    [
        (
            {
                "sequence_id": "seqA",
                " Sequence ": "seqB",
                "time_s": 0.0,
                "x_m": 1.0,
                "y_m": 2.0,
                "z_m": 3.0,
            },
            "sequence_id",
        ),
        (
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "X": 1.0,
                " x ": 9.0,
                "y_m": 2.0,
                "z_m": 3.0,
            },
            "x_m",
        ),
    ],
)
def test_load_submission_csv_rejects_ambiguous_alias_headers(
    tmp_path: Path,
    row: dict[str, object],
    canonical: str,
) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match=rf"ambiguous columns for '{canonical}'"):
        load_submission_csv(path)
