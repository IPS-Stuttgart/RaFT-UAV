from __future__ import annotations

from raft_uav.mmuad.submission import (
    load_official_track5_results_frame,
    load_official_track5_template_file,
)


_OFFICIAL_CSV = """Sequence,Timestamp,Position,Classification
001,0.0,"(1,2,3)",2
010,1.0,"(4,5,6)",3
"""


def test_official_track5_csv_loader_preserves_zero_padded_sequence_ids(tmp_path) -> None:
    path = tmp_path / "mmaud_results.csv"
    path.write_text(_OFFICIAL_CSV, encoding="utf-8")

    frame = load_official_track5_results_frame(path)

    assert frame["Sequence"].tolist() == ["001", "010"]


def test_official_track5_template_loader_preserves_zero_padded_sequence_ids(
    tmp_path,
) -> None:
    path = tmp_path / "template.csv"
    path.write_text(_OFFICIAL_CSV, encoding="utf-8")

    frame = load_official_track5_template_file(path)

    assert frame["sequence_id"].tolist() == ["001", "010"]
