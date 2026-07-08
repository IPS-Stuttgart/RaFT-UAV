from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_text_cli import (
    _read_csv_preserving_sequence_id as read_estimate_csv,
)
from raft_uav.mmuad.track5_sequence_gate_fit_text_cli import (
    _read_csv_preserving_sequence_id as read_gate_csv,
)


def _write_normalized_rows(path: Path) -> None:
    pd.DataFrame(
        {
            "sequence_id": ["NA", "N/A"],
            "time_s": [0.0, 1.0],
            "state_x_m": [1.0, 2.0],
            "state_y_m": [2.0, 3.0],
            "state_z_m": [3.0, 4.0],
        }
    ).to_csv(path, index=False)


def test_estimate_fit_wrapper_preserves_default_null_tokens_as_sequence_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "normalized.csv"
    _write_normalized_rows(csv_path)

    rows = read_estimate_csv(csv_path)

    assert rows["sequence_id"].tolist() == ["NA", "N/A"]


def test_estimate_fit_wrapper_strips_header_whitespace(tmp_path: Path) -> None:
    csv_path = tmp_path / "normalized.csv"
    csv_path.write_text(
        " sequence_id , time_s , state_x_m , state_y_m , state_z_m \n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )

    rows = read_estimate_csv(csv_path)

    assert rows.columns.tolist() == [
        "sequence_id",
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
    ]
    assert rows.loc[0, "sequence_id"] == "001"


def test_sequence_gate_fit_wrapper_preserves_default_null_tokens_as_sequence_ids(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "normalized.csv"
    _write_normalized_rows(csv_path)

    rows = read_gate_csv(csv_path)

    assert rows["sequence_id"].tolist() == ["NA", "N/A"]
