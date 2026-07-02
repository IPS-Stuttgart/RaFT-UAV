from __future__ import annotations

from pathlib import Path

from raft_uav.multi_uav_lts.duplicate_audit import audit_duplicate_predictions


def test_lts_duplicate_audit_preserves_large_integer_key_precision(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    first_id = 9_007_199_254_740_992
    second_id = first_id + 1
    rows = [
        f"1,{first_id},10,20,5,6,0.9,1,1",
        f"1,{second_id},10,20,5,6,0.9,1,1",
    ]
    prediction_dir.joinpath("S_00.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    audit = audit_duplicate_predictions(prediction_dir)

    assert audit.clean
    assert audit.parse_errors == 0
    assert audit.duplicate_rows == 0
    assert audit.files[0].row_count == 2


def test_lts_duplicate_audit_accepts_integer_like_decimal_and_scientific_ids(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("S_00.txt").write_text(
        "4.0,7e0,10,20,5,6,0.9,1,1\n"
        "4,7,10,20,5,6,0.9,1,1\n",
        encoding="utf-8",
    )

    audit = audit_duplicate_predictions(prediction_dir)

    assert not audit.clean
    assert audit.parse_errors == 0
    assert audit.duplicate_rows == 1
    assert audit.duplicate_keys[0].frame_id == 4
    assert audit.duplicate_keys[0].object_id == 7
