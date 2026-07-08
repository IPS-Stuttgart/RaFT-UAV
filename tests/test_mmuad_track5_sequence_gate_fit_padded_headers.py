from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad import track5_sequence_gate_fit as _impl
from raft_uav.mmuad.track5_sequence_gate_fit_text_cli import _read_csv_preserving_sequence_id


def test_sequence_gate_fit_wrapper_strips_padded_sequence_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "padded.csv"
    pd.DataFrame({" Sequence ": ["001"], "value": [4]}).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path)

    assert rows.columns.tolist() == ["Sequence", "value"]
    assert rows.loc[0, "Sequence"] == "001"


def test_sequence_gate_fit_loader_accepts_padded_normalized_headers_via_wrapper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "normalized_padded.csv"
    csv_path.write_text(
        " sequence_id , time_s , x_m , y_m , z_m \n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_impl.pd, "read_csv", _read_csv_preserving_sequence_id)

    rows = _impl._load_track5_gate_rows(csv_path)

    assert rows.columns.tolist() == [
        "sequence_id",
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
    ]
    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "time_s"] == 0.0
