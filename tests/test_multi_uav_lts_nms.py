from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

from raft_uav.multi_uav_lts.nms import apply_lts_nms_to_text
from raft_uav.multi_uav_lts.nms import main as nms_main
from raft_uav.multi_uav_lts.nms import repair_lts_submission_with_nms


def _prediction_text() -> str:
    return "\n".join(
        [
            "1,1,0,0,10,10,0.95,1,1",
            "1,2,1,1,10,10,0.80,1,1",
            "1,3,1,1,10,10,0.70,2,1",
            "2,1,100,100,10,10,0.90,1,1",
        ]
    ) + "\n"


def test_lts_nms_suppresses_same_class_overlap_but_keeps_other_class() -> None:
    repaired, summary = apply_lts_nms_to_text(
        _prediction_text(),
        iou_threshold=0.5,
        class_aware=True,
    )

    assert summary.input_rows == 4
    assert summary.output_rows == 3
    assert summary.suppressed_rows == 1
    assert "1,1,0,0,10,10,0.95,1,1" in repaired
    assert "1,2,1,1,10,10,0.8,1,1" not in repaired
    assert "1,3,1,1,10,10,0.7,2,1" in repaired


def test_lts_nms_repair_writes_valid_template_zip(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "pred"
    prediction_dir.mkdir()
    (prediction_dir / "seq1.txt").write_text(_prediction_text(), encoding="utf-8")
    template_zip = tmp_path / "template.zip"
    with ZipFile(template_zip, "w") as archive:
        archive.writestr("seq1.txt", "")
    output_dir = tmp_path / "out"
    output_zip = tmp_path / "nms.zip"

    summary = repair_lts_submission_with_nms(
        prediction_dir,
        output_dir,
        output_zip=output_zip,
        template_zip=template_zip,
        iou_threshold=0.5,
        class_aware=True,
        require_valid=True,
    )

    assert summary.valid_zip is True
    assert summary.suppressed_rows == 1
    assert output_zip.exists()
    assert (output_dir / "multi_uav_lts_nms_summary.json").exists()
    payload = json.loads((output_dir / "multi_uav_lts_nms_summary.json").read_text())
    assert payload["output_rows"] == 3
    with ZipFile(output_zip) as archive:
        assert archive.namelist() == ["seq1.txt"]


def test_lts_nms_cli_writes_artifacts(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "pred"
    prediction_dir.mkdir()
    (prediction_dir / "seq1.txt").write_text(_prediction_text(), encoding="utf-8")
    template_zip = tmp_path / "template.zip"
    with ZipFile(template_zip, "w") as archive:
        archive.writestr("seq1.txt", "")
    output_dir = tmp_path / "out"
    output_zip = tmp_path / "nms.zip"

    status = nms_main(
        [
            str(prediction_dir),
            "--output-dir",
            str(output_dir),
            "--output-zip",
            str(output_zip),
            "--template-zip",
            str(template_zip),
            "--iou-threshold",
            "0.5",
            "--require-valid",
        ]
    )

    assert status == 0
    assert output_zip.exists()
    assert (output_dir / "multi_uav_lts_nms_file_summary.csv").exists()


def test_lts_nms_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-multi-uav-lts-nms"]
        == "raft_uav.multi_uav_lts.nms:main"
    )
