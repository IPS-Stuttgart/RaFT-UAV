from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.duplicate_audit import audit_duplicate_predictions


def _row(frame: int, object_id: int) -> str:
    return f"{frame},{object_id},10,20,5,6,0.9,1,1\n"


def test_duplicate_prediction_audit_rejects_zero_frame_or_object_ids(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("S_00.txt").write_text(
        _row(0, 1) + _row(1, 0),
        encoding="utf-8",
    )

    audit = audit_duplicate_predictions(prediction_dir)

    assert not audit.clean
    assert audit.total_rows == 2
    assert audit.parse_errors == 2
    assert audit.duplicate_key_count == 0
    assert audit.duplicate_rows == 0
