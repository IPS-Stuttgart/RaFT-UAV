from __future__ import annotations

import pytest

from raft_uav.mmuad.submission import load_sequence_class_map


def test_csv_class_map_rejects_conflicting_duplicate_sequence_rows(tmp_path) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    class_map_csv.write_text(
        "sequence_id,uav_type\n001,2\n 001 ,3\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="class-map CSV assigns conflicting UAV types to sequence '001'",
    ):
        load_sequence_class_map(class_map_csv)


def test_csv_class_map_allows_repeated_identical_sequence_rows(tmp_path) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    class_map_csv.write_text(
        "sequence_id,uav_type\n001,2\n 001 ,2\n",
        encoding="utf-8",
    )

    assert load_sequence_class_map(class_map_csv) == {"001": "2"}
