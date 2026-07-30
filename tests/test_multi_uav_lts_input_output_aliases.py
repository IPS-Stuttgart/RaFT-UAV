from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.cli import (
    package_submission,
    write_constant_first_frame_predictions,
    write_first_frame_labels,
)


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_package_submission_rejects_template_output_alias_without_modifying_template(
    tmp_path: Path,
) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    (prediction_dir / "S_00.txt").write_text(
        "1,1,10,20,5,6,1,1,1\n",
        encoding="utf-8",
    )
    template = tmp_path / "template.zip"
    _write_zip(template, {"S_00.txt": ""})
    original = template.read_bytes()

    with pytest.raises(ValueError, match="output ZIP must differ from template ZIP"):
        package_submission(
            prediction_dir,
            template,
            template_zip=template,
        )

    assert template.read_bytes() == original


def test_constant_first_frame_rejects_label_output_alias_without_modifying_labels(
    tmp_path: Path,
) -> None:
    sequence_root = tmp_path / "sequences"
    sequence_dir = sequence_root / "S_00"
    sequence_dir.mkdir(parents=True)
    (sequence_dir / "00000.jpg").write_bytes(b"image")
    labels = tmp_path / "labels"
    labels.mkdir()
    label_path = labels / "S_00.txt"
    original = "1,7,10,20,5,6,1,1,1\n"
    label_path.write_text(original, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="output directory must differ from first-frame label directory",
    ):
        write_constant_first_frame_predictions(sequence_root, labels, labels)

    assert label_path.read_text(encoding="utf-8") == original


def test_first_frame_labels_rejects_truth_output_alias_without_modifying_truth(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth"
    truth.mkdir()
    truth_path = truth / "S_00.txt"
    original = (
        "1,7,10,20,5,6,1,1,1\n"
        "2,7,11,20,5,6,1,1,1\n"
    )
    truth_path.write_text(original, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="output directory must differ from truth directory",
    ):
        write_first_frame_labels(truth, truth)

    assert truth_path.read_text(encoding="utf-8") == original
