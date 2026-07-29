from __future__ import annotations

import math
import zipfile
from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.metrics import evaluate_lts_predictions


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_perfect_predictions_score_one(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    rows = "1,1,0,0,10,10,1,1,1\n2,1,1,0,10,10,1,1,1\n"
    _write(truth / "S.txt", rows)
    _write(predictions / "S.txt", rows)

    metrics = evaluate_lts_predictions(predictions, truth)

    assert metrics.codabench_hota == pytest.approx(1.0)
    assert metrics.codabench_mota == pytest.approx(1.0)
    assert metrics.codabench_idf1 == pytest.approx(1.0)
    assert metrics.hota == pytest.approx(1.0)
    assert metrics.deta == pytest.approx(1.0)
    assert metrics.assa == pytest.approx(1.0)
    assert metrics.loca == pytest.approx(1.0)
    assert metrics.mota == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)


def test_id_switch_reduces_association_metrics(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n2,1,1,0,10,10,1,1,1\n")
    _write(predictions / "S.txt", "1,5,0,0,10,10,1,1,1\n2,6,1,0,10,10,1,1,1\n")

    metrics = evaluate_lts_predictions(predictions, truth)

    assert metrics.deta == pytest.approx(1.0)
    assert metrics.assa == pytest.approx(0.5)
    assert metrics.codabench_hota == pytest.approx(math.sqrt(0.5))
    assert metrics.codabench_mota == pytest.approx(0.5)
    assert metrics.codabench_idf1 == pytest.approx(0.5)
    assert metrics.hota == pytest.approx(math.sqrt(0.5))
    assert metrics.mota == pytest.approx(0.5)
    assert metrics.idf1 == pytest.approx(0.5)
    assert metrics.id_switches == 1


def test_hota_uses_multiple_localization_thresholds(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    # IoU = 50 / 150 = 1/3.
    _write(predictions / "S.txt", "1,1,5,0,10,10,1,1,1\n")

    metrics = evaluate_lts_predictions(predictions, truth)

    matched = [alpha <= (1.0 / 3.0 + 1e-12) for alpha in metrics.alphas]
    assert metrics.hota_true_positives == tuple(1 if value else 0 for value in matched)
    assert metrics.codabench_hota == pytest.approx(1.0)
    assert metrics.combined_hota_at_005 == pytest.approx(1.0)
    assert metrics.hota == pytest.approx(sum(matched) / len(matched))


def test_combination_is_detection_weighted(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write(truth / "A.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(predictions / "A.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(truth / "B.txt", "1,1,0,0,10,10,1,1,1\n2,1,0,0,10,10,1,1,1\n3,1,0,0,10,10,1,1,1\n")
    _write(predictions / "B.txt", "")

    metrics = evaluate_lts_predictions(predictions, truth)

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 3
    assert metrics.mota == pytest.approx(0.25)
    assert metrics.deta == pytest.approx(0.25)
    assert metrics.codabench_hota == pytest.approx(0.5)
    assert metrics.codabench_mota == pytest.approx((1.0 + 0.0 + 0.25) / 3.0)


def test_prediction_zip_is_supported(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("S.txt", "1,9,0,0,10,10,1,1,1\n")

    metrics = evaluate_lts_predictions(archive, truth)

    assert metrics.hota == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)


def test_duplicate_frame_object_keys_are_rejected(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    predictions = tmp_path / "predictions"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(
        predictions / "S.txt",
        "1,2,0,0,10,10,1,1,1\n1,2,1,0,10,10,1,1,1\n",
    )

    with pytest.raises(ValueError, match="duplicate predictions"):
        evaluate_lts_predictions(predictions, truth)
