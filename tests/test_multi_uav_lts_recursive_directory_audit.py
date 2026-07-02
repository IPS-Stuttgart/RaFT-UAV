from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.duplicate_audit import audit_duplicate_predictions


def _row(frame: int, object_id: int) -> str:
    return f"{frame},{object_id},10,20,5,6,0.9,1,1\n"


def test_lts_audit_reads_nested_directory_prediction_files(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    nested_dir = prediction_dir / "nested"
    nested_dir.mkdir(parents=True)
    nested_dir.joinpath("S_00.txt").write_text(_row(1, 1) + _row(1, 1), encoding="utf-8")

    audit = audit_duplicate_predictions(prediction_dir)

    assert audit.file_count == 1
    assert audit.files[0].name == "nested/S_00.txt"
    assert audit.duplicate_rows == 1
