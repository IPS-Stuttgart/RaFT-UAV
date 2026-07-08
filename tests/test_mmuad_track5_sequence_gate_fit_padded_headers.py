from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_sequence_gate_fit_text_cli import _read_csv_preserving_sequence_id


def test_sequence_gate_fit_wrapper_strips_padded_sequence_header_preserving_id(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "padded.csv"
    pd.DataFrame({" Sequence ": ["001"], " value ": [4]}).to_csv(csv_path, index=False)

    rows = _read_csv_preserving_sequence_id(csv_path)

    assert rows.columns.tolist() == ["Sequence", "value"]
    assert rows.loc[0, "Sequence"] == "001"
