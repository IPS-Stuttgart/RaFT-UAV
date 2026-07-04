from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.submission import load_sequence_class_map


def test_csv_class_map_preserves_textual_sequence_ids(tmp_path: Path) -> None:
    path = tmp_path / "class_map.csv"
    path.write_text("sequence_id,uav_type\n001,2\n010,3\n", encoding="utf-8")

    assert load_sequence_class_map(path) == {"001": "2", "010": "3"}
