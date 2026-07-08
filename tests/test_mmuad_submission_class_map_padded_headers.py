from __future__ import annotations

from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.submission import load_sequence_class_map


def test_csv_class_map_accepts_padded_alias_headers(tmp_path) -> None:
    class_map_csv = tmp_path / "class_map.csv"
    class_map_csv.write_text(
        " Sequence , Type \n001,2\n010,3\n",
        encoding="utf-8",
    )

    class_map = load_sequence_class_map(class_map_csv)

    assert class_map == {"001": "2", "010": "3"}


def test_official_track5_template_accepts_padded_alias_headers(tmp_path) -> None:
    template_csv = tmp_path / "template.csv"
    template_csv.write_text(
        ' Sequence , Timestamp , Position , Classification \n'
        '001,0.0,"(0,0,0)",2\n'
        '010,1.5,"(1,2,3)",3\n',
        encoding="utf-8",
    )

    template = load_official_track5_template_file(template_csv)

    assert template.to_dict("records") == [
        {"sequence_id": "001", "time_s": 0.0},
        {"sequence_id": "010", "time_s": 1.5},
    ]
