from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.run_multi_uav_lts_official_baseline import _validate_inputs


def _paths(tmp_path: Path, *, template_zip: Path | None) -> dict[str, Path]:
    return {
        "sequence_root": tmp_path / "missing-images",
        "first_frame_label_dir": tmp_path / "missing-first-frame-labels",
        "template_zip": template_zip,
        "botsort_root": tmp_path / "missing-botsort",
    }


def test_package_only_validation_skips_inference_only_inputs(tmp_path: Path) -> None:
    template_zip = tmp_path / "submission.zip"
    with zipfile.ZipFile(template_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("S_00.txt", "")

    _validate_inputs(
        _paths(tmp_path, template_zip=template_zip),
        require_inference_inputs=False,
        require_botsort=False,
    )


def test_inference_validation_still_requires_sequence_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="sequence_root"):
        _validate_inputs(
            _paths(tmp_path, template_zip=None),
            require_inference_inputs=True,
            require_botsort=False,
        )
