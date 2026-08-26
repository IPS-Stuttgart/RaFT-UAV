from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.fixed_population import postprocess_fixed_population


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_replaces_existing_prediction_set(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "S.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(predictions / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(output / "stale.txt", "1,99,0,0,1,1,1,1,1\n")
    _write(output / "notes.json", "{}")

    postprocess_fixed_population(predictions, labels, output)

    assert sorted(path.name for path in output.glob("*.txt")) == ["S.txt"]
    assert (output / "notes.json").read_text(encoding="utf-8") == "{}"


def test_validation_failure_preserves_existing_outputs(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "A.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(labels / "B.txt", "2,8,0,0,10,10,1,1,1\n")
    _write(predictions / "A.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(output / "old.txt", "original\n")

    with pytest.raises(ValueError, match="first-frame-only"):
        postprocess_fixed_population(predictions, labels, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "original\n"
    assert not (output / "A.txt").exists()


def test_unknown_prediction_sequence_preserves_existing_outputs(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(labels / "A.txt", "1,7,0,0,10,10,1,1,1\n")
    _write(predictions / "A.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(predictions / "B.txt", "1,2,0,0,10,10,1,1,1\n")
    _write(output / "old.txt", "original\n")

    with pytest.raises(
        ValueError,
        match=r"prediction input contains unknown sequence files: B\.txt",
    ):
        postprocess_fixed_population(predictions, labels, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "original\n"
    assert not (output / "A.txt").exists()


@pytest.mark.parametrize(
    ("label_state", "error_type", "message"),
    [
        ("missing", FileNotFoundError, "first-frame label directory does not exist"),
        ("file", NotADirectoryError, "first-frame label path is not a directory"),
        ("empty", ValueError, r"first-frame label directory contains no \.txt files"),
    ],
)
def test_invalid_label_input_preserves_existing_outputs(
    tmp_path: Path,
    label_state: str,
    error_type: type[Exception],
    message: str,
) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    output = tmp_path / "output"
    _write(predictions / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(output / "old.txt", "original\n")
    if label_state == "file":
        labels.write_text("not a directory\n", encoding="utf-8")
    elif label_state == "empty":
        labels.mkdir()

    with pytest.raises(error_type, match=message):
        postprocess_fixed_population(predictions, labels, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize(
    ("output_alias", "message"),
    [
        ("labels", "output directory must differ from first-frame label directory"),
        ("predictions", "output directory must differ from prediction directory"),
    ],
)
def test_rejects_output_directory_that_aliases_an_input(
    tmp_path: Path,
    output_alias: str,
    message: str,
) -> None:
    labels = tmp_path / "labels"
    predictions = tmp_path / "predictions"
    label_text = "1,7,0,0,10,10,1,1,1\n"
    prediction_text = "1,1,0,0,10,10,1,1,1\n"
    _write(labels / "S.txt", label_text)
    _write(predictions / "S.txt", prediction_text)
    output = labels if output_alias == "labels" else predictions

    with pytest.raises(ValueError, match=message):
        postprocess_fixed_population(predictions, labels, output)

    assert (labels / "S.txt").read_text(encoding="utf-8") == label_text
    assert (predictions / "S.txt").read_text(encoding="utf-8") == prediction_text
