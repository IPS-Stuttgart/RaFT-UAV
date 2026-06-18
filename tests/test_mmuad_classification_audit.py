from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.classification_audit import (
    OFFICIAL_TRACK5_CLASS_NAMES,
    build_audit_from_files,
    build_mmuad_classification_audit,
    write_mmuad_classification_audit,
)


def _official_rows(classes_by_sequence: dict[str, int]) -> pd.DataFrame:
    rows = []
    for sequence, class_id in classes_by_sequence.items():
        for idx in range(2):
            rows.append(
                {
                    "Sequence": sequence,
                    "Timestamp": 1000.0 + idx,
                    "Position": f"({idx},{idx + 1},{idx + 2})",
                    "Classification": class_id,
                }
            )
    return pd.DataFrame(rows)


def test_classification_audit_detects_constant_default_label() -> None:
    truth = _official_rows({"seq0001": 0, "seq0002": 1, "seq0003": 2, "seq0004": 2})
    results = truth.copy()
    results["Classification"] = 0

    audit = build_mmuad_classification_audit(truth=truth, results=results)

    assert audit.summary["current_accuracy"] == 0.25
    assert audit.summary["submission_constant_prediction"] == "0"
    assert audit.summary["constant_prediction_accuracy"] == 0.25
    assert audit.summary["default_label_explains_score"] is True
    assert audit.summary["majority_baseline_prediction"] == "2"
    assert audit.summary["majority_baseline_accuracy"] == 0.5

    rows = audit.classification_audit.set_index("sequence")
    assert rows.loc["seq0001", "class_source"] == "default_or_constant_submission_label"
    assert bool(rows.loc["seq0001", "valid_class_mapping"]) is True
    assert bool(rows.loc["seq0001", "valid_submission_label"]) is True
    assert rows.loc["seq0001", "truth_class_values"] == "0"
    assert rows.loc["seq0001", "predicted_class_values"] == "0"
    assert rows.loc["seq0001", "truth_majority_class"] == "0"
    assert rows.loc["seq0001", "predicted_majority_class"] == "0"
    assert rows.loc["seq0001", "class_string"] == OFFICIAL_TRACK5_CLASS_NAMES[0]
    assert rows.loc["seq0002", "per_sequence_accuracy"] == 0.0
    assert rows.loc["seq0002", "sequence_accuracy"] == 0.0
    assert rows.loc["seq0001", "constant_class_0_accuracy"] == 0.25
    assert rows.loc["seq0001", "constant_class_1_accuracy"] == 0.25
    assert rows.loc["seq0001", "constant_class_2_accuracy"] == 0.5
    assert rows.loc["seq0001", "constant_class_3_accuracy"] == 0.0
    assert rows.loc["seq0001", "global_majority_class"] == "2"
    assert rows.loc["seq0001", "global_majority_accuracy"] == 0.5
    assert rows.loc["seq0001", "per_sequence_oracle_majority_accuracy"] == 1.0

    confusion = audit.confusion_matrix.set_index(["ground_truth_class", "predicted_class"])
    assert confusion.loc[("0", "0"), "count"] == 2
    assert confusion.loc[("1", "0"), "count"] == 2
    assert confusion.loc[("2", "0"), "count"] == 4


def test_classification_audit_writes_requested_csvs(tmp_path: Path) -> None:
    truth = _official_rows({"seq0001": 0, "seq0002": 3})
    results = truth.copy()
    results.loc[results["Sequence"] == "seq0002", "Classification"] = 2
    audit = build_mmuad_classification_audit(truth=truth, results=results)

    paths = write_mmuad_classification_audit(audit, output_dir=tmp_path)

    assert Path(paths["classification_audit_csv"]).name == "mmuad_classification_audit.csv"
    assert Path(paths["confusion_matrix_csv"]).name == "mmuad_class_confusion_matrix.csv"
    assert Path(paths["sequence_class_summary_csv"]).name == "mmuad_sequence_class_summary.csv"
    for path in paths.values():
        assert Path(path).exists()

    audit_rows = pd.read_csv(paths["classification_audit_csv"])
    assert {
        "sequence",
        "row_count",
        "truth_class_values",
        "predicted_class_values",
        "truth_majority_class",
        "predicted_majority_class",
        "class_source",
        "class_map_key",
        "class_map_value",
        "valid_submission_label",
        "sequence_accuracy",
    }.issubset(audit_rows.columns)


def test_classification_audit_reads_official_files(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.csv"
    results_path = tmp_path / "mmaud_results.csv"
    _official_rows({"seq0001": 0, "seq0002": 1}).to_csv(truth_path, index=False)
    _official_rows({"seq0001": 0, "seq0002": 0}).to_csv(results_path, index=False)

    audit = build_audit_from_files(truth_path=truth_path, results_path=results_path)

    assert audit.summary["truth_unique_classes"] == ["0", "1"]
    assert audit.summary["predicted_unique_classes"] == ["0"]
    assert audit.sequence_class_summary["training_labels_available"].eq(False).all()


def test_classification_audit_reports_class_map_source() -> None:
    truth = _official_rows({"seq0001": 0, "seq0002": 1})
    results = truth.copy()
    results.loc[results["Sequence"] == "seq0002", "Classification"] = 0

    audit = build_mmuad_classification_audit(
        truth=truth,
        results=results,
        class_map={"seq0001": "0", "seq0002": "1"},
    )

    rows = audit.classification_audit.set_index("sequence")
    assert rows.loc["seq0001", "class_source"] == "class_map_sequence_label"
    assert rows.loc["seq0001", "class_map_key"] == "seq0001"
    assert rows.loc["seq0001", "class_map_value"] == "0"
    assert (
        rows.loc["seq0002", "class_source"]
        == "class_map_sequence_label_ignored_or_remapped"
    )
    assert audit.summary["class_map_sequence_count"] == 2
