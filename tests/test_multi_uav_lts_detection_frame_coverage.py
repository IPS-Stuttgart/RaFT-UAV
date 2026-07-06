from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.coverage_audit import audit_prediction_coverage
from raft_uav.multi_uav_lts.coverage_audit import main as coverage_audit_main


def _row(frame: int, object_id: int = 1) -> str:
    return f"{frame},{object_id},10,20,5,6,0.9,1,1\n"


def _sequence_root(tmp_path: Path) -> Path:
    root = tmp_path / "TestImages"
    seq = root / "A_00"
    seq.mkdir(parents=True)
    for frame in range(1, 5):
        seq.joinpath(f"{frame:06d}.jpg").write_text("", encoding="utf-8")
    return root


def test_detection_frame_fraction_is_reported_when_sequence_root_is_available(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("A_00.txt").write_text(
        _row(1, 1) + _row(3, 2),
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(prediction_dir, sequence_root=_sequence_root(tmp_path))
    row = audit.rows[0]

    assert audit.ready
    assert row.expected_frame_count == 4
    assert row.detected_frame_count == 2
    assert row.detection_frame_fraction == pytest.approx(0.5)
    assert audit.detection_frame_fraction_min == pytest.approx(0.5)
    assert audit.detection_frame_fraction_mean == pytest.approx(0.5)


def test_min_detection_frame_fraction_can_block_sparse_outputs(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("A_00.txt").write_text(_row(1), encoding="utf-8")

    audit = audit_prediction_coverage(
        prediction_dir,
        sequence_root=_sequence_root(tmp_path),
        min_detection_frame_fraction=0.5,
    )
    row = audit.rows[0]

    assert not audit.ready
    assert audit.low_detection_coverage_file_count == 1
    assert audit.low_detection_coverage_files == ["A_00.txt"]
    assert audit.blocking_reasons == ["low_detection_coverage_files"]
    assert audit.status_counts == {"low_coverage": 1}
    assert row.status == "low_coverage"
    assert row.detection_frame_fraction == pytest.approx(0.25)


def test_low_detection_frame_fraction_gate_exits_nonzero(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_dir.joinpath("A_00.txt").write_text(_row(1), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        coverage_audit_main(
            [
                str(prediction_dir),
                "--sequence-root",
                str(_sequence_root(tmp_path)),
                "--min-detection-frame-fraction",
                "0.5",
                "--require-ready",
            ]
        )

    assert exc_info.value.code == 1
