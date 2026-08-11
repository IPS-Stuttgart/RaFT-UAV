from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd

from raft_uav.mmuad.track5_estimate_text_cli import (
    _read_csv_preserving_sequence_id as read_estimate_csv,
)
from raft_uav.mmuad.track5_sequence_gate_fit_text_cli import (
    _read_csv_preserving_sequence_id as read_gate_csv,
)


def test_estimate_wrapper_scalar_dtype_is_warning_free(tmp_path: Path) -> None:
    csv_path = tmp_path / "estimate.csv"
    csv_path.write_text(
        "sequence_id,time_s,state_x_m\n001,1.25,3.5\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        rows = read_estimate_csv(csv_path, dtype=float)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == 1.25
    assert rows.loc[0, "state_x_m"] == 3.5


def test_gate_wrapper_scalar_dtype_is_warning_free_for_padded_id_header(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "gate.csv"
    csv_path.write_text(
        " sequence_id ,time_s,value\n001,1.25,4\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        rows = read_gate_csv(csv_path, dtype=str)

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == "1.25"
    assert rows.loc[0, "value"] == "4"
