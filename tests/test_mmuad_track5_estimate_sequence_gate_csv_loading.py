from __future__ import annotations

from raft_uav.mmuad.track5_estimate_sequence_gate import _read_csv_preserve_text


def test_sequence_gate_cli_csv_loader_preserves_text_sequence_ids(tmp_path) -> None:
    csv_path = tmp_path / "weights.csv"
    csv_path.write_text("Sequence,gate_weight\nseq001,0.75\nseq010,0.25\n", encoding="utf-8")

    rows = _read_csv_preserve_text(csv_path)

    assert list(rows.columns) == ["Sequence", "gate_weight"]
    assert rows.loc[0, "Sequence"] == "seq001"
    assert rows.loc[1, "Sequence"] == "seq010"
