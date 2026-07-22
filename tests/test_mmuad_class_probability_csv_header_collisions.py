from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.class_probability_csv import read_class_probability_csv


def test_class_probability_csv_rejects_case_insensitive_header_collisions(
    tmp_path: Path,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        "sequence_id,Predicted_Probability_0,PREDICTED_PROBABILITY_0\n"
        "001,0.1,0.9\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="ambiguous columns after trimming whitespace and ignoring case",
    ) as exc_info:
        read_class_probability_csv(probabilities_csv)

    message = str(exc_info.value)
    assert "Predicted_Probability_0" in message
    assert "PREDICTED_PROBABILITY_0" in message
