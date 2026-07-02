from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.coverage_audit import (
    audit_prediction_coverage,
    main as coverage_audit_main,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_prediction_coverage_audit_unsorted_rows_are_not_ready(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(template_zip, {"A_00.txt": ""})
    prediction_dir.joinpath("A_00.txt").write_text(
        "2,1,10,20,5,6,0.9,1,1\n1,2,10,20,5,6,0.9,1,1\n",
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(prediction_dir, template_zip=template_zip)

    assert not audit.ready
    assert audit.unsorted_rows == 1
    assert audit.rows[0].status == "unsorted"

    with pytest.raises(SystemExit) as exc_info:
        coverage_audit_main(
            [
                str(prediction_dir),
                "--template-zip",
                str(template_zip),
                "--require-ready",
            ]
        )

    assert exc_info.value.code == 1
