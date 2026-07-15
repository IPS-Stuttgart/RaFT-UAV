from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.class_probability_csv import read_class_probability_csv


@pytest.mark.parametrize(
    ("header", "duplicate_name"),
    [
        (
            "sequence_id, sequence_id ,predicted_probability_0",
            "sequence_id",
        ),
        (
            "sequence_id,predicted_probability_0, predicted_probability_0 ",
            "predicted_probability_0",
        ),
    ],
)
def test_class_probability_csv_rejects_whitespace_collapsed_duplicate_headers(
    tmp_path: Path,
    header: str,
    duplicate_name: str,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        f"{header}\n001,002,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"ambiguous columns.*{duplicate_name}",
    ):
        read_class_probability_csv(probabilities_csv)
