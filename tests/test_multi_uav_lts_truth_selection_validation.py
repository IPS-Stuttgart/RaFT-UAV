from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.cli import score_lts_predictions

_ROW = "1,1,10,10,10,10,1,1,1\n"


def _write_sequence(directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath(f"{name}.txt").write_text(_ROW, encoding="utf-8")


def test_score_rejects_empty_truth_directory(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    truth.mkdir()
    _write_sequence(predictions, "S_00")

    with pytest.raises(ValueError, match="contains no .txt sequence files"):
        score_lts_predictions(predictions, truth)


def test_score_rejects_unknown_requested_truth_sequence(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write_sequence(truth, "S_00")
    _write_sequence(predictions, "S_00")

    with pytest.raises(ValueError, match="requested truth sequences are unavailable: S_01"):
        score_lts_predictions(predictions, truth, sequences=["S_01"])


def test_score_accepts_available_requested_truth_sequence(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write_sequence(truth, "S_00")
    _write_sequence(predictions, "S_00")

    scorecard = score_lts_predictions(predictions, truth, sequences=["S_00"])

    assert scorecard.sequence_count == 1
    assert scorecard.matches == 1
