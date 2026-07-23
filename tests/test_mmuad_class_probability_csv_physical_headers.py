from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.class_probability_csv import read_class_probability_csv


def test_class_probability_csv_rejects_exact_duplicate_physical_headers(
    tmp_path: Path,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        "sequence_id,sequence_id,predicted_probability_0\n"
        "canonical-001,conflicting-001,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"ambiguous columns after trimming whitespace and ignoring case: 'sequence_id'",
    ):
        read_class_probability_csv(probabilities_csv)
