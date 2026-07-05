from __future__ import annotations

from raft_uav.mmuad.submission import load_sequence_class_map


def test_csv_map_preserves_zero_padded_ids(tmp_path) -> None:
    path = tmp_path / "map.csv"
    path.write_text(
        "sequence_id,uav_type\n001,2\n010,3\n,1\n020,\n",
        encoding="utf-8",
    )

    assert load_sequence_class_map(path) == {"001": "2", "010": "3"}
