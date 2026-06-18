"""Classification diagnostics for public MMUAD/UG2+ Track 5 tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import load_official_track5_results_frame


OFFICIAL_TRACK5_CLASS_NAMES = {
    0: "Mavic 3",
    1: "M30",
    2: "M300",
    3: "Phantom 4",
}
OFFICIAL_TRACK5_CLASS_IDS = frozenset(OFFICIAL_TRACK5_CLASS_NAMES)


@dataclass(frozen=True)
class MmuadClassificationAudit:
    """Frames emitted by the MMUAD Track 5 classification audit."""

    classification_audit: pd.DataFrame
    confusion_matrix: pd.DataFrame
    sequence_class_summary: pd.DataFrame
    summary: dict[str, Any]


def build_mmuad_classification_audit(
    *,
    truth: pd.DataFrame,
    results: pd.DataFrame,
    training_truth: pd.DataFrame | None = None,
    class_names: dict[int, str] | None = None,
) -> MmuadClassificationAudit:
    """Compare official Track 5 truth and result classifications.

    The public validation reference exposes integer ``Classification`` values.
    This audit validates that both truth and predictions stay in the expected
    four-class ID domain and makes constant-default submissions obvious.
    """

    class_names = dict(class_names or OFFICIAL_TRACK5_CLASS_NAMES)
    truth_rows = _class_rows(truth)
    result_rows = _class_rows(results)
    training_rows = _class_rows(training_truth) if training_truth is not None else None

    truth_majority = _majority_class(truth_rows["Classification"])
    result_constant = _constant_class(result_rows["Classification"])
    training_majority = (
        _majority_class(training_rows["Classification"]) if training_rows is not None else None
    )

    truth_by_sequence = _sequence_class_stats(truth_rows, prefix="truth")
    result_by_sequence = _sequence_class_stats(result_rows, prefix="predicted")
    sequences = sorted(set(truth_by_sequence).union(result_by_sequence))

    sequence_records: list[dict[str, Any]] = []
    for sequence in sequences:
        truth_stats = truth_by_sequence.get(sequence, {})
        result_stats = result_by_sequence.get(sequence, {})
        truth_class = truth_stats.get("truth_class")
        predicted_class = result_stats.get("predicted_class")
        predicted_rows = int(result_stats.get("predicted_row_count", 0) or 0)
        correct_rows = _sequence_correct_count(result_rows, sequence, truth_class)
        per_sequence_accuracy = (
            float(correct_rows / predicted_rows) if predicted_rows > 0 else np.nan
        )
        class_source = _class_source(result_stats)
        valid_class_mapping = _valid_class_id(truth_class) and _valid_class_id(predicted_class)
        majority_correct = (
            bool(_classes_equal(truth_class, truth_majority)) if truth_class is not None else False
        )
        training_majority_correct = (
            bool(_classes_equal(truth_class, training_majority))
            if training_majority is not None and truth_class is not None
            else None
        )
        row = {
            "sequence": sequence,
            "ground_truth_class": _class_text(truth_class),
            "predicted_class": _class_text(predicted_class),
            "class_source": class_source,
            "class_id": _class_text(predicted_class),
            "class_string": _class_string(predicted_class, class_names),
            "valid_class_mapping": bool(valid_class_mapping),
            "per_sequence_accuracy": per_sequence_accuracy,
            "majority_baseline_prediction": _class_text(truth_majority),
            "majority_baseline_class_string": _class_string(truth_majority, class_names),
            "majority_baseline_correct": majority_correct,
            "training_labels_available": training_rows is not None,
            "training_majority_prediction": _class_text(training_majority),
            "training_majority_class_string": _class_string(training_majority, class_names),
            "training_majority_correct": training_majority_correct,
            "truth_row_count": int(truth_stats.get("truth_row_count", 0) or 0),
            "predicted_row_count": predicted_rows,
            "correct_row_count": int(correct_rows),
            "truth_unique_class_count": int(truth_stats.get("truth_unique_class_count", 0) or 0),
            "predicted_unique_class_count": int(
                result_stats.get("predicted_unique_class_count", 0) or 0
            ),
            "truth_class_string": _class_string(truth_class, class_names),
            "predicted_class_string": _class_string(predicted_class, class_names),
            "truth_class_counts": str(truth_stats.get("truth_class_counts", "")),
            "predicted_class_counts": str(result_stats.get("predicted_class_counts", "")),
            "submission_constant_prediction": _class_text(result_constant),
            "default_label_explains_score": None,
        }
        sequence_records.append(row)

    sequence_summary = pd.DataFrame(sequence_records)
    current_accuracy = _row_accuracy_from_sequence_truth(result_rows, truth_by_sequence)
    constant_accuracy = (
        _constant_prediction_accuracy(truth_rows, result_constant) if result_constant is not None else np.nan
    )
    majority_accuracy = _constant_prediction_accuracy(truth_rows, truth_majority)
    training_majority_accuracy = (
        _constant_prediction_accuracy(truth_rows, training_majority)
        if training_majority is not None
        else np.nan
    )
    default_explains = (
        result_constant is not None
        and np.isfinite(current_accuracy)
        and np.isfinite(constant_accuracy)
        and abs(float(current_accuracy) - float(constant_accuracy)) <= 1.0e-12
    )
    if not sequence_summary.empty:
        sequence_summary["default_label_explains_score"] = bool(default_explains)

    audit_columns = [
        "sequence",
        "ground_truth_class",
        "predicted_class",
        "class_source",
        "class_id",
        "class_string",
        "valid_class_mapping",
        "per_sequence_accuracy",
        "majority_baseline_prediction",
    ]
    audit = sequence_summary[audit_columns].copy()
    confusion = _confusion_matrix_rows(
        result_rows,
        truth_by_sequence,
        class_names=class_names,
    )
    summary = {
        "truth_row_count": int(len(truth_rows)),
        "predicted_row_count": int(len(result_rows)),
        "sequence_count": int(len(sequence_summary)),
        "truth_unique_classes": _sorted_class_texts(truth_rows["Classification"]),
        "predicted_unique_classes": _sorted_class_texts(result_rows["Classification"]),
        "official_expected_class_ids": sorted(OFFICIAL_TRACK5_CLASS_IDS),
        "official_expected_class_strings": [
            class_names[class_id] for class_id in sorted(OFFICIAL_TRACK5_CLASS_IDS)
        ],
        "valid_truth_class_mapping": bool(_valid_class_series(truth_rows["Classification"])),
        "valid_predicted_class_mapping": bool(_valid_class_series(result_rows["Classification"])),
        "current_accuracy": float(current_accuracy),
        "submission_constant_prediction": _class_text(result_constant),
        "submission_constant_class_string": _class_string(result_constant, class_names),
        "constant_prediction_accuracy": float(constant_accuracy),
        "default_label_explains_score": bool(default_explains),
        "majority_baseline_prediction": _class_text(truth_majority),
        "majority_baseline_class_string": _class_string(truth_majority, class_names),
        "majority_baseline_accuracy": float(majority_accuracy),
        "training_labels_available": training_rows is not None,
        "training_majority_prediction": _class_text(training_majority),
        "training_majority_class_string": _class_string(training_majority, class_names),
        "training_majority_accuracy_on_truth": (
            float(training_majority_accuracy)
            if np.isfinite(training_majority_accuracy)
            else None
        ),
    }
    return MmuadClassificationAudit(
        classification_audit=audit,
        confusion_matrix=confusion,
        sequence_class_summary=sequence_summary,
        summary=summary,
    )


def write_mmuad_classification_audit(
    audit: MmuadClassificationAudit,
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write the standard MMUAD classification audit CSV artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "classification_audit_csv": output_dir / "mmuad_classification_audit.csv",
        "confusion_matrix_csv": output_dir / "mmuad_class_confusion_matrix.csv",
        "sequence_class_summary_csv": output_dir / "mmuad_sequence_class_summary.csv",
    }
    audit.classification_audit.to_csv(paths["classification_audit_csv"], index=False)
    audit.confusion_matrix.to_csv(paths["confusion_matrix_csv"], index=False)
    audit.sequence_class_summary.to_csv(paths["sequence_class_summary_csv"], index=False)
    return {key: str(value) for key, value in paths.items()}


def build_audit_from_files(
    *,
    truth_path: Path,
    results_path: Path,
    training_truth_path: Path | None = None,
) -> MmuadClassificationAudit:
    """Load official Track 5 files and build the classification audit."""

    training_truth = (
        load_official_track5_results_frame(training_truth_path)
        if training_truth_path is not None
        else None
    )
    return build_mmuad_classification_audit(
        truth=load_official_track5_results_frame(truth_path),
        results=load_official_track5_results_frame(results_path),
        training_truth=training_truth,
    )


def _class_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["Sequence", "Timestamp", "Classification"])
    missing = {"Sequence", "Timestamp", "Classification"}.difference(frame.columns)
    if missing:
        raise ValueError(f"classification audit rows missing columns: {sorted(missing)}")
    rows = frame[["Sequence", "Timestamp", "Classification"]].copy()
    rows["Sequence"] = rows["Sequence"].astype(str)
    rows["Classification"] = rows["Classification"].map(_normal_class_value)
    return rows.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _normal_class_value(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        text = str(value).strip()
        return text or None
    if np.isfinite(number) and number.is_integer():
        return int(number)
    text = str(value).strip()
    return text or None


def _sequence_class_stats(rows: pd.DataFrame, *, prefix: str) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for sequence, group in rows.groupby("Sequence", sort=True):
        values = group["Classification"].dropna()
        counts = values.value_counts(dropna=False)
        class_id = _majority_class(values)
        stats[str(sequence)] = {
            f"{prefix}_class": class_id,
            f"{prefix}_row_count": int(len(group)),
            f"{prefix}_unique_class_count": int(values.nunique(dropna=False)),
            f"{prefix}_class_counts": _format_class_counts(counts),
        }
    return stats


def _sequence_correct_count(
    result_rows: pd.DataFrame,
    sequence: str,
    truth_class: int | str | None,
) -> int:
    if truth_class is None:
        return 0
    predicted = result_rows.loc[result_rows["Sequence"].astype(str) == str(sequence), "Classification"]
    return int(sum(_classes_equal(value, truth_class) for value in predicted))


def _row_accuracy_from_sequence_truth(
    result_rows: pd.DataFrame,
    truth_by_sequence: dict[str, dict[str, Any]],
) -> float:
    total = 0
    correct = 0
    for _, row in result_rows.iterrows():
        sequence = str(row["Sequence"])
        truth_class = truth_by_sequence.get(sequence, {}).get("truth_class")
        if truth_class is None:
            continue
        total += 1
        correct += int(_classes_equal(row["Classification"], truth_class))
    return float(correct / total) if total else np.nan


def _constant_prediction_accuracy(rows: pd.DataFrame, prediction: int | str | None) -> float:
    if prediction is None or rows.empty:
        return np.nan
    truth_by_sequence = _sequence_class_stats(rows, prefix="truth")
    total = 0
    correct = 0
    for _, row in rows.iterrows():
        truth_class = truth_by_sequence.get(str(row["Sequence"]), {}).get("truth_class")
        if truth_class is None:
            continue
        total += 1
        correct += int(_classes_equal(prediction, truth_class))
    return float(correct / total) if total else np.nan


def _confusion_matrix_rows(
    result_rows: pd.DataFrame,
    truth_by_sequence: dict[str, dict[str, Any]],
    *,
    class_names: dict[int, str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if result_rows.empty:
        return pd.DataFrame(
            columns=[
                "ground_truth_class",
                "predicted_class",
                "count",
                "ground_truth_class_string",
                "predicted_class_string",
                "sequence_count",
            ]
        )
    rows = result_rows.copy()
    rows["ground_truth_class"] = [
        truth_by_sequence.get(str(sequence), {}).get("truth_class")
        for sequence in rows["Sequence"]
    ]
    grouped = (
        rows.groupby(["ground_truth_class", "Classification"], dropna=False)["Sequence"]
        .agg(row_count="count", sequence_count=lambda values: int(values.astype(str).nunique()))
        .reset_index()
    )
    for _, row in grouped.iterrows():
        truth_class = _normal_class_value(row["ground_truth_class"])
        predicted_class = _normal_class_value(row["Classification"])
        records.append(
            {
                "ground_truth_class": _class_text(truth_class),
                "predicted_class": _class_text(predicted_class),
                "count": int(row["row_count"]),
                "ground_truth_class_string": _class_string(truth_class, class_names),
                "predicted_class_string": _class_string(predicted_class, class_names),
                "sequence_count": int(row["sequence_count"]),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["ground_truth_class", "predicted_class"]
    ).reset_index(drop=True)


def _majority_class(values: pd.Series) -> int | str | None:
    clean = [value for value in values.dropna().tolist() if value is not None]
    if not clean:
        return None
    counts = pd.Series(clean, dtype=object).value_counts(dropna=False)
    return sorted(counts.items(), key=lambda item: (-int(item[1]), _class_text(item[0])))[0][0]


def _constant_class(values: pd.Series) -> int | str | None:
    clean = [value for value in values.dropna().tolist() if value is not None]
    if not clean:
        return None
    unique = {_class_text(value): value for value in clean}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _valid_class_series(values: pd.Series) -> bool:
    return all(_valid_class_id(value) for value in values.dropna().tolist())


def _valid_class_id(value: Any) -> bool:
    return isinstance(value, int) and value in OFFICIAL_TRACK5_CLASS_IDS


def _classes_equal(left: Any, right: Any) -> bool:
    return _class_text(left) == _class_text(right)


def _class_string(value: Any, class_names: dict[int, str]) -> str:
    if isinstance(value, int):
        return class_names.get(value, "")
    return ""


def _class_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isfinite(value) and value.is_integer():
        return str(int(value))
    return str(value)


def _format_class_counts(counts: pd.Series) -> str:
    return ";".join(f"{_class_text(label)}:{int(count)}" for label, count in counts.items())


def _sorted_class_texts(values: pd.Series) -> list[str]:
    return sorted({_class_text(value) for value in values.dropna().tolist()})


def _class_source(result_stats: dict[str, Any]) -> str:
    row_count = int(result_stats.get("predicted_row_count", 0) or 0)
    unique_count = int(result_stats.get("predicted_unique_class_count", 0) or 0)
    if row_count == 0:
        return "missing_prediction"
    if unique_count == 1:
        return "submission_constant_label"
    return "submission_row_labels"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-classification-audit",
        description="audit public MMUAD/UG2+ Track 5 Classification labels",
    )
    parser.add_argument("--truth", type=Path, required=True, help="official truth CSV/ZIP")
    parser.add_argument("--results", type=Path, required=True, help="official results CSV/ZIP")
    parser.add_argument(
        "--training-truth",
        type=Path,
        help="optional official training truth/class labels for a training-majority baseline",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    audit = build_audit_from_files(
        truth_path=args.truth,
        results_path=args.results,
        training_truth_path=args.training_truth,
    )
    paths = write_mmuad_classification_audit(audit, output_dir=args.output_dir)
    print("mmuad_classification_audit=ok")
    for key, value in paths.items():
        print(f"{key}={value}")
    for key in (
        "current_accuracy",
        "submission_constant_prediction",
        "default_label_explains_score",
        "majority_baseline_prediction",
        "majority_baseline_accuracy",
        "training_labels_available",
    ):
        print(f"{key}={audit.summary[key]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
