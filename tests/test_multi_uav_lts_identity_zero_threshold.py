from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.metrics import evaluate_lts_predictions


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_zero_threshold_rejects_zero_iou_identity_pairs(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(predictions / "S.txt", "1,7,20,0,10,10,1,1,1\n")

    metrics = evaluate_lts_predictions(
        predictions,
        truth,
        clear_iou_threshold=0.0,
    )

    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.id_true_positives == 0
    assert metrics.id_false_positives == 1
    assert metrics.id_false_negatives == 1
    assert metrics.idf1 == pytest.approx(0.0)


def test_zero_threshold_keeps_positive_iou_identity_pairs(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(predictions / "S.txt", "1,7,9,0,10,10,1,1,1\n")

    metrics = evaluate_lts_predictions(
        predictions,
        truth,
        clear_iou_threshold=0.0,
    )

    assert metrics.id_true_positives == 1
    assert metrics.id_false_positives == 0
    assert metrics.id_false_negatives == 0
    assert metrics.idf1 == pytest.approx(1.0)
