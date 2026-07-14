from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.classification_audit import build_mmuad_classification_audit


def _official_rows(classes_by_sequence: dict[str, int]) -> pd.DataFrame:
    rows = []
    for sequence, class_id in classes_by_sequence.items():
        for index in range(2):
            rows.append(
                {
                    "Sequence": sequence,
                    "Timestamp": 1000.0 + index,
                    "Position": "(0,0,0)",
                    "Classification": class_id,
                }
            )
    return pd.DataFrame(rows)


def test_classification_audit_counts_missing_truth_sequences_as_incorrect() -> None:
    truth = _official_rows({"seq0001": 0, "seq0002": 1})
    results = _official_rows({"seq0001": 0})

    audit = build_mmuad_classification_audit(truth=truth, results=results)

    assert audit.summary["current_accuracy"] == 0.5
    assert audit.summary["constant_prediction_accuracy"] == 0.5
    assert audit.summary["default_label_explains_score"] is True
    assert audit.summary["classification_matched_truth_row_count"] == 2
    assert audit.summary["classification_missing_truth_row_count"] == 2
    assert audit.summary["classification_matched_sequence_count"] == 1
    assert audit.summary["classification_missing_sequence_count"] == 1
    assert audit.summary["classification_coverage_fraction"] == 0.5
