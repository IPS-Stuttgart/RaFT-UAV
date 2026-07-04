from __future__ import annotations

import zipfile
from pathlib import Path

from raft_uav.multi_uav_lts.coverage_audit import audit_prediction_coverage


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_coverage_audit_ignores_template_non_prediction_entries(tmp_path: Path) -> None:
    template_zip = tmp_path / "template.zip"
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_zip(
        template_zip,
        {
            "A_00.txt": "",
            "template_manifest.csv": "sequence,count\nA_00,1\n",
        },
    )
    prediction_dir.joinpath("A_00.txt").write_text(
        "1,1,10,20,5,6,0.9,1,1\n",
        encoding="utf-8",
    )

    audit = audit_prediction_coverage(prediction_dir, template_zip=template_zip)

    assert audit.ready
    assert audit.expected_file_count == 1
    assert audit.missing_files == []
    assert audit.extra_files == []
    assert [row.name for row in audit.rows] == ["A_00.txt"]
