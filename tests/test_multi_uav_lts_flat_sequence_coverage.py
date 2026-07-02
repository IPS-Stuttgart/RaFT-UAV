from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.coverage_audit import audit_prediction_coverage


def test_coverage_audit_accepts_flat_image_root(tmp_path: Path) -> None:
    sequence_root = tmp_path / "FlatSequence"
    output_dir = tmp_path / "outputs"
    sequence_root.mkdir()
    output_dir.mkdir()
    for frame in range(1, 4):
        sequence_root.joinpath(f"{frame:06d}.jpg").write_text("", encoding="utf-8")
    output_dir.joinpath("FlatSequence.txt").write_text(
        "2,1,10,20,5,6,0.9,1,1\n",
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(output_dir, sequence_root=sequence_root)

    assert audit.ready
    assert audit.expected_file_count == 1
    assert audit.missing_files == []
    assert audit.extra_files == []
    assert audit.rows[0].name == "FlatSequence.txt"
    assert audit.rows[0].expected_frame_count == 3
    assert audit.rows[0].status == "ok"
