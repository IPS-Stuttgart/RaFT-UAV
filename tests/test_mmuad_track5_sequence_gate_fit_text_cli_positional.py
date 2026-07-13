from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.track5_sequence_gate_fit_text_cli import _read_csv_preserving_sequence_id


@pytest.mark.parametrize(
    "parser_options",
    [
        {"converters": {0: int, 1: int}},
        {"dtype": {0: int, 1: int}},
    ],
)
def test_sequence_gate_fit_wrapper_protects_positional_sequence_parser_options(
    tmp_path: Path,
    parser_options: dict,
) -> None:
    csv_path = tmp_path / "normalized.csv"
    csv_path.write_text("sequence_id,value\n001,4\n", encoding="utf-8")

    rows = _read_csv_preserving_sequence_id(csv_path, **parser_options)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "value"] == 4


def test_sequence_gate_fit_wrapper_protects_padded_header_with_scalar_dtype(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "normalized.csv"
    csv_path.write_text(" Sequence ,value\n001,4\n", encoding="utf-8")

    rows = _read_csv_preserving_sequence_id(csv_path, dtype=float)

    assert rows.loc[0, "Sequence"] == "001"
    assert rows.loc[0, "value"] == 4.0
