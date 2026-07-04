from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_diagnostics import (
    build_classification_diagnostics,
    main as diagnostics_main,
)


def _evaluation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002", "seq0002", "seq0003"],
            "timestamp": [0.0, 1.0, 0.0, 1.0, 0.0],
            "matched": [True, True, True, True, False],
            "predicted_uav_type": ["0", "1", "2", "2", "3"],
            "truth_uav_type": ["0", "0", "2", "3", "3"],
            "uav_type_correct": [True, False, True, False, False],
        }
    )


def _official_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0, 1.0],
            "Position": ["(0, 0, 0)", "(1, 0, 0)", "(2, 0, 0)", "(3, 0, 0)"],
            "Classification": [0, 1, 2, 2],
        }
    )


def _official_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0, 1.0],
            "Position": ["(0, 0, 0)", "(1, 0, 0)", "(2, 0, 0)", "(3, 0, 0)"],
            "Classification": [0, 0, 2, 3],
        }
    )


def test_classification_diagnostics_build_by_sequence_and_confusion() -> None:
    by_sequence, confusion, summary = build_classification_diagnostics(_evaluation_rows())

    seq1 = by_sequence.loc[by_sequence["sequence_id"] == "seq0001"].iloc[0]
    seq2 = by_sequence.loc[by_sequence["sequence_id"] == "seq0002"].iloc[0]
    assert seq1["classification_accuracy"] == pytest.approx(0.5)
    assert seq1["truth_uav_type"] == "0"
    assert seq1["predicted_type_count"] == 2
    assert seq2["classification_accuracy"] == pytest.approx(0.5)
    assert summary["matched_count"] == 4
    assert summary["correct_count"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)

    confusion_lookup = {
        (row.truth_uav_type, row.predicted_uav_type): row.count
        for row in confusion.itertuples(index=False)
    }
    assert confusion_lookup[("0", "0")] == 1
    assert confusion_lookup[("0", "1")] == 1
    assert confusion_lookup[("2", "2")] == 1
    assert confusion_lookup[("3", "2")] == 1


def test_classification_diagnostics_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_classification_diagnostics(pd.DataFrame({"sequence_id": ["seq0001"]}))


def test_classification_diagnostics_cli_scores_official_zip(tmp_path: Path) -> None:
    results_csv = tmp_path / "mmaud_results.csv"
    truth_csv = tmp_path / "truth.csv"
    results_zip = tmp_path / "submission.zip"
    by_sequence_csv = tmp_path / "classification_by_sequence.csv"
    confusion_csv = tmp_path / "classification_confusion.csv"
    summary_json = tmp_path / "classification_summary.json"
    _official_results().to_csv(results_csv, index=False)
    _official_truth().to_csv(truth_csv, index=False)
    with ZipFile(results_zip, "w") as archive:
        archive.write(results_csv, arcname="mmaud_results.csv")

    status = diagnostics_main(
        [
            "--results",
            str(results_zip),
            "--truth",
            str(truth_csv),
            "--by-sequence-csv",
            str(by_sequence_csv),
            "--confusion-csv",
            str(confusion_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert status == 0
    assert by_sequence_csv.exists()
    assert confusion_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["matched_count"] == 4
    assert payload["accuracy"] == pytest.approx(0.5)
    assert pd.read_csv(by_sequence_csv)["sequence_id"].tolist() == ["seq0001", "seq0002"]
