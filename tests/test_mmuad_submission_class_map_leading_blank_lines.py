from __future__ import annotations

import pytest

from raft_uav.mmuad.submission import load_sequence_class_map


@pytest.mark.parametrize("prefix", ["\n", "\n   \n"])
def test_csv_class_map_rejects_duplicate_header_after_leading_blank_lines(
    tmp_path,
    prefix: str,
) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    class_map_csv.write_text(
        prefix
        + "sequence_id,sequence_id,uav_type\n"
        + "001,999,2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="class-map CSV has ambiguous columns after trimming whitespace and ignoring case",
    ):
        load_sequence_class_map(class_map_csv)


def test_csv_class_map_accepts_unique_header_after_leading_blank_lines(tmp_path) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    class_map_csv.write_text(
        "\n   \nsequence_id,uav_type\n001,2\n",
        encoding="utf-8",
    )

    assert load_sequence_class_map(class_map_csv) == {"001": "2"}
