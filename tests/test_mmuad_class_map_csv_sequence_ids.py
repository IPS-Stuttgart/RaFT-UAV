from __future__ import annotations

from raft_uav.mmuad.submission import load_sequence_class_map


def test_csv_class_map_preserves_textual_sequence_ids(tmp_path) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    class_map_csv.write_text(
        "Sequence,uav_type\n001,2\n010,3\n,1\n",
        encoding="utf-8",
    )

    class_map = load_sequence_class_map(class_map_csv)

    assert class_map == {"001": "2", "010": "3"}
