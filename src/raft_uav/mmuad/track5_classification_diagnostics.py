"""Classification diagnostics for MMUAD/UG2+ Track 5 result rows.

The main Track 5 scorecard reports pooled UAV type accuracy.  For method
improvement and paper tables it is also useful to know which sequences and
classes fail.  This module builds per-sequence accuracy rows and a compact
truth-vs-prediction confusion table from the local evaluator output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_evaluation_truth_file,
    load_mmaud_results_file,
)
from raft_uav.mmuad.schema import load_jsonable

_CLASS_LABELS = ("0", "1", "2", "3")


def build_classification_diagnostics(
    evaluation_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return by-sequence, confusion, and summary diagnostics.

    The input is normally the ``rows`` table returned by
    ``evaluate_mmaud_results(..., metric_protocol='public-track5')``.  Only
    matched rows with a finite truth class are used for accuracy metrics.
    """

    rows = pd.DataFrame(evaluation_rows).copy()
    if rows.empty:
        return _empty_by_sequence(), _empty_confusion(), _empty_summary()
    required = {"sequence_id", "matched", "predicted_uav_type", "truth_uav_type"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"classification diagnostics missing columns: {sorted(missing)}")
    matched = rows.loc[_bool_series(rows["matched"])].copy()
    matched["predicted_uav_type"] = matched["predicted_uav_type"].map(_class_label)
    matched["truth_uav_type"] = matched["truth_uav_type"].map(_class_label)
    matched = matched.loc[matched["truth_uav_type"].notna()].copy()
    if matched.empty:
        return _empty_by_sequence(), _empty_confusion(), _empty_summary()
    matched["uav_type_correct"] = matched["predicted_uav_type"] == matched["truth_uav_type"]
    by_sequence = _by_sequence_table(matched)
    confusion = _confusion_table(matched)
    summary = _summary_payload(matched, by_sequence=by_sequence, confusion=confusion)
    return by_sequence, confusion, summary


def write_classification_diagnostics(
    *,
    by_sequence: pd.DataFrame,
    confusion: pd.DataFrame,
    summary: dict[str, Any],
    by_sequence_csv: Path | None = None,
    confusion_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, str]:
    """Write diagnostics artifacts and return path labels."""

    paths: dict[str, str] = {}
    if by_sequence_csv is not None:
        by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
        by_sequence.to_csv(by_sequence_csv, index=False)
        paths["classification_by_sequence_csv"] = str(by_sequence_csv)
    if confusion_csv is not None:
        confusion_csv.parent.mkdir(parents=True, exist_ok=True)
        confusion.to_csv(confusion_csv, index=False)
        paths["classification_confusion_csv"] = str(confusion_csv)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(load_jsonable(summary), indent=2), encoding="utf-8")
        paths["classification_summary_json"] = str(summary_json)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-classification-diagnostics",
        description="write MMUAD Track 5 classification by-sequence and confusion diagnostics",
    )
    parser.add_argument("--results", type=Path, required=True, help="mmaud_results.csv or ZIP")
    parser.add_argument("--truth", type=Path, required=True, help="normalized or official truth CSV/ZIP")
    parser.add_argument("--class-map", type=Path, help="optional sequence-to-class map")
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--by-sequence-csv", type=Path, required=True)
    parser.add_argument("--confusion-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args(argv)

    results = load_mmaud_results_file(args.results)
    truth = load_evaluation_truth_file(args.truth)
    evaluation = evaluate_mmaud_results(
        results,
        truth,
        metric_protocol="public-track5",
        timestamp_tolerance_s=float(args.timestamp_tolerance_s),
        class_map_path=args.class_map,
    )
    by_sequence, confusion, summary = build_classification_diagnostics(evaluation["rows"])
    summary.update(
        {
            "results_path": str(args.results),
            "truth_path": str(args.truth),
            "class_map_path": str(args.class_map) if args.class_map is not None else None,
            "timestamp_tolerance_s": float(args.timestamp_tolerance_s),
        }
    )
    written = write_classification_diagnostics(
        by_sequence=by_sequence,
        confusion=confusion,
        summary=summary,
        by_sequence_csv=args.by_sequence_csv,
        confusion_csv=args.confusion_csv,
        summary_json=args.summary_json,
    )
    print("track5_classification_diagnostics=ok")
    for name, path in written.items():
        print(f"{name}={path}")
    print(f"classification_accuracy={summary.get('accuracy')}")
    print(f"matched_count={summary.get('matched_count')}")
    return 0


def _by_sequence_table(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby(rows["sequence_id"].astype(str), sort=True):
        truth_counts = group["truth_uav_type"].value_counts()
        pred_counts = group["predicted_uav_type"].value_counts()
        correct = _bool_series(group["uav_type_correct"])
        records.append(
            {
                "sequence_id": str(sequence_id),
                "row_count": int(len(group)),
                "classification_accuracy": _safe_fraction(int(correct.sum()), int(len(group))),
                "correct_count": int(correct.sum()),
                "incorrect_count": int(len(group) - int(correct.sum())),
                "truth_uav_type": _dominant_label(truth_counts),
                "predicted_uav_type": _dominant_label(pred_counts),
                "truth_type_count": int(len(truth_counts)),
                "predicted_type_count": int(len(pred_counts)),
            }
        )
    return pd.DataFrame.from_records(records, columns=_by_sequence_columns())


def _confusion_table(rows: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(
        set(_CLASS_LABELS)
        | {str(value) for value in rows["truth_uav_type"].dropna().unique()}
        | {str(value) for value in rows["predicted_uav_type"].dropna().unique()}
    )
    total = int(len(rows))
    records: list[dict[str, Any]] = []
    for truth_label in labels:
        truth_group = rows.loc[rows["truth_uav_type"] == truth_label]
        truth_total = int(len(truth_group))
        for predicted_label in labels:
            count = int((truth_group["predicted_uav_type"] == predicted_label).sum())
            records.append(
                {
                    "truth_uav_type": truth_label,
                    "predicted_uav_type": predicted_label,
                    "count": count,
                    "fraction_of_all": _safe_fraction(count, total),
                    "recall_contribution": _safe_fraction(count, truth_total),
                }
            )
    return pd.DataFrame.from_records(records)


def _summary_payload(
    rows: pd.DataFrame,
    *,
    by_sequence: pd.DataFrame,
    confusion: pd.DataFrame,
) -> dict[str, Any]:
    correct = _bool_series(rows["uav_type_correct"])
    class_records: dict[str, dict[str, Any]] = {}
    for truth_label, group in rows.groupby("truth_uav_type", sort=True):
        group_correct = _bool_series(group["uav_type_correct"])
        class_records[str(truth_label)] = {
            "support": int(len(group)),
            "accuracy": _safe_fraction(int(group_correct.sum()), int(len(group))),
            "correct_count": int(group_correct.sum()),
            "incorrect_count": int(len(group) - int(group_correct.sum())),
        }
    worst_sequences = by_sequence.sort_values(
        ["classification_accuracy", "row_count", "sequence_id"],
        ascending=[True, False, True],
    ).head(10)
    return {
        "schema": "raft-uav-mmuad-track5-classification-diagnostics-v1",
        "matched_count": int(len(rows)),
        "correct_count": int(correct.sum()),
        "incorrect_count": int(len(rows) - int(correct.sum())),
        "accuracy": _safe_fraction(int(correct.sum()), int(len(rows))),
        "sequence_count": int(by_sequence["sequence_id"].nunique()) if not by_sequence.empty else 0,
        "confusion_rows": int(len(confusion)),
        "class_metrics": class_records,
        "worst_sequences": worst_sequences.to_dict(orient="records"),
    }


def _class_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _dominant_label(counts: pd.Series) -> str | None:
    if counts.empty:
        return None
    return str(counts.index[0])


def _bool_series(values: Any) -> pd.Series:
    series = pd.Series(values)
    if series.empty:
        return series.astype(bool)
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return series.map(lambda value: str(value).strip().lower() in {"1", "true", "yes"})


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _empty_by_sequence() -> pd.DataFrame:
    return pd.DataFrame(columns=_by_sequence_columns())


def _empty_confusion() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "truth_uav_type",
            "predicted_uav_type",
            "count",
            "fraction_of_all",
            "recall_contribution",
        ]
    )


def _empty_summary() -> dict[str, Any]:
    return {
        "schema": "raft-uav-mmuad-track5-classification-diagnostics-v1",
        "matched_count": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "accuracy": None,
        "sequence_count": 0,
        "confusion_rows": 0,
        "class_metrics": {},
        "worst_sequences": [],
    }


def _by_sequence_columns() -> list[str]:
    return [
        "sequence_id",
        "row_count",
        "classification_accuracy",
        "correct_count",
        "incorrect_count",
        "truth_uav_type",
        "predicted_uav_type",
        "truth_type_count",
        "predicted_type_count",
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
