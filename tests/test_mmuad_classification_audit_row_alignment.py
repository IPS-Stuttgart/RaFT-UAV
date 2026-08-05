from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.classification_audit import build_mmuad_classification_audit


def _rows(entries: list[tuple[str, object, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        entries,
        columns=["Sequence", "Timestamp", "Classification"],
    )


def test_classification_accuracy_penalizes_missing_result_rows() -> None:
    truth = _rows(
        [
            ("seq-a", 0.0, 0),
            ("seq-a", 1.0, 0),
            ("seq-b", 0.0, 1),
            ("seq-b", 1.0, 1),
        ]
    )
    results = _rows([("seq-a", 0.0, 0)])

    audit = build_mmuad_classification_audit(truth=truth, results=results)

    assert audit.summary["current_accuracy"] == pytest.approx(0.25)
    assert audit.summary["classification_scored_row_count"] == 4
    assert audit.summary["matched_row_count"] == 1
    assert audit.summary["missing_prediction_row_count"] == 3
    assert audit.summary["extra_prediction_row_count"] == 0
    assert audit.summary["row_key_parity"] is False

    by_sequence = audit.sequence_class_summary.set_index("sequence")
    assert by_sequence.loc["seq-a", "per_sequence_accuracy"] == pytest.approx(0.5)
    assert by_sequence.loc["seq-b", "per_sequence_accuracy"] == pytest.approx(0.0)
    assert by_sequence.loc["seq-b", "missing_prediction_row_count"] == 2

    confusion = audit.confusion_matrix.set_index(
        ["ground_truth_class", "predicted_class"]
    )
    assert confusion.loc[("0", "0"), "count"] == 1
    assert confusion.loc[("0", "<missing-row>"), "count"] == 1
    assert confusion.loc[("1", "<missing-row>"), "count"] == 2


def test_classification_accuracy_penalizes_duplicate_and_foreign_rows() -> None:
    truth = _rows(
        [
            ("seq-a", 0.0, 0),
            ("seq-a", 1.0, 0),
        ]
    )
    results = _rows(
        [
            ("seq-a", 0.0, 0),
            ("seq-a", 0.0, 0),
            ("seq-a", 1.0, 0),
            ("seq-x", 0.0, 0),
        ]
    )

    audit = build_mmuad_classification_audit(truth=truth, results=results)

    assert audit.summary["current_accuracy"] == pytest.approx(0.5)
    assert audit.summary["classification_scored_row_count"] == 4
    assert audit.summary["matched_row_count"] == 2
    assert audit.summary["extra_prediction_row_count"] == 2

    by_sequence = audit.sequence_class_summary.set_index("sequence")
    assert by_sequence.loc["seq-a", "per_sequence_accuracy"] == pytest.approx(2.0 / 3.0)
    assert by_sequence.loc["seq-a", "extra_prediction_row_count"] == 1
    assert by_sequence.loc["seq-x", "per_sequence_accuracy"] == pytest.approx(0.0)


def test_invalid_timestamps_never_match_across_truth_and_results() -> None:
    truth = _rows([("seq-a", None, 0)])
    results = _rows([("seq-a", None, 0)])

    audit = build_mmuad_classification_audit(truth=truth, results=results)

    assert audit.summary["current_accuracy"] == pytest.approx(0.0)
    assert audit.summary["classification_scored_row_count"] == 2
    assert audit.summary["matched_row_count"] == 0
    assert audit.summary["missing_prediction_row_count"] == 1
    assert audit.summary["extra_prediction_row_count"] == 1
