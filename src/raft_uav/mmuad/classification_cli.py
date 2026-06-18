"""CLI for MMUAD sequence-level UAV type baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.classification import (
    SEQUENCE_CLASSIFIER_METHODS,
    SEQUENCE_CLASSIFIER_LOSO_PREDICTION_COLUMNS,
    apply_sequence_loso_labels_to_submission,
    build_sequence_classifier_loso_predictions,
    classify_sequences_from_features,
    load_sequence_class_labels,
    sequence_features_from_files,
    write_sequence_classifier_loso_predictions,
    write_sequence_classification_result,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-sequence-classifier",
        description=(
            "train a simple sequence-level UAV type classifier and write a "
            "sequence_id,uav_type class map for Track 5 submission exports"
        ),
    )
    parser.add_argument(
        "--train-feature-table",
        type=Path,
        action="append",
        help="CSV/JSON table with training sequence rows; may be repeated",
    )
    parser.add_argument(
        "--predict-feature-table",
        type=Path,
        action="append",
        help="CSV/JSON table with target sequence rows; may be repeated",
    )
    parser.add_argument(
        "--train-labels",
        type=Path,
        help="sequence class-map CSV/JSON/YAML or official Track 5 truth CSV/ZIP",
    )
    parser.add_argument(
        "--eval-labels",
        type=Path,
        help="optional class-map/truth labels for reporting target accuracy",
    )
    parser.add_argument(
        "--method",
        choices=SEQUENCE_CLASSIFIER_METHODS,
        default="nearest-neighbor",
    )
    parser.add_argument("--k", type=int, default=1, help="nearest-neighbor vote count")
    parser.add_argument("--output-class-map", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--train-features-csv", type=Path)
    parser.add_argument("--predict-features-csv", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument(
        "--loso-eval",
        action="store_true",
        help="run leave-one-sequence-out diagnostics from one reference and feature table",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="official Track 5 reference CSV/ZIP or class-map labels for --loso-eval",
    )
    parser.add_argument(
        "--selected-tracklets",
        type=Path,
        action="append",
        help="selected tracklet/candidate table for --loso-eval; may be repeated",
    )
    parser.add_argument(
        "--loso-predictions-csv",
        type=Path,
        help="output mmuad_sequence_classifier_loso_predictions.csv path",
    )
    parser.add_argument("--submission-in", type=Path, help="optional existing results CSV/ZIP")
    parser.add_argument(
        "--submission-out",
        type=Path,
        help="optional relabeled results CSV/ZIP that preserves pose columns",
    )
    args = parser.parse_args(argv)

    if args.loso_eval:
        return _run_loso_eval(args, parser)

    _require_args(
        parser,
        args,
        [
            ("train_feature_table", "--train-feature-table"),
            ("predict_feature_table", "--predict-feature-table"),
            ("train_labels", "--train-labels"),
            ("output_class_map", "--output-class-map"),
        ],
    )
    train_features = sequence_features_from_files(args.train_feature_table)
    predict_features = sequence_features_from_files(args.predict_feature_table)
    train_labels = load_sequence_class_labels(args.train_labels)
    eval_labels = load_sequence_class_labels(args.eval_labels) if args.eval_labels is not None else None
    result = classify_sequences_from_features(
        train_features=train_features,
        train_labels=train_labels,
        predict_features=predict_features,
        method=args.method,
        k=args.k,
        eval_labels=eval_labels,
    )
    paths = write_sequence_classification_result(
        result,
        output_class_map=args.output_class_map,
        predictions_csv=args.predictions_csv,
        train_features_csv=args.train_features_csv,
        predict_features_csv=args.predict_features_csv,
        metrics_json=args.metrics_json,
    )
    print("mmuad_sequence_classifier=ok")
    for key, value in paths.items():
        print(f"{key}={value}")
    for key in (
        "method",
        "train_sequence_count",
        "predict_sequence_count",
        "feature_count",
        "labels_available",
        "sequence_accuracy",
    ):
        if key in result.metrics:
            print(f"{key}={result.metrics[key]}")
    return 0


def _run_loso_eval(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _require_args(
        parser,
        args,
        [
            ("reference", "--reference"),
            ("selected_tracklets", "--selected-tracklets"),
        ],
    )
    if (args.submission_in is None) != (args.submission_out is None):
        parser.error("--submission-in and --submission-out must be provided together")
    features = sequence_features_from_files(args.selected_tracklets)
    labels = load_sequence_class_labels(args.reference)
    predictions = build_sequence_classifier_loso_predictions(
        features=features,
        labels=labels,
        method=args.method,
    )
    predictions_csv = args.loso_predictions_csv
    if predictions_csv is None:
        output_dir = (
            args.submission_out.parent
            if args.submission_out is not None
            else Path.cwd()
        )
        predictions_csv = output_dir / "mmuad_sequence_classifier_loso_predictions.csv"
    write_sequence_classifier_loso_predictions(predictions, predictions_csv)
    print("mmuad_sequence_classifier_loso=ok")
    print(f"loso_predictions_csv={predictions_csv}")
    print(f"sequence_count={len(predictions)}")
    print(f"sequence_accuracy={float(predictions['correct'].mean())}")
    print(f"columns={','.join(SEQUENCE_CLASSIFIER_LOSO_PREDICTION_COLUMNS)}")
    if args.submission_in is not None and args.submission_out is not None:
        output = apply_sequence_loso_labels_to_submission(
            submission_in=args.submission_in,
            loso_predictions=predictions,
            submission_out=args.submission_out,
        )
        print(f"submission_out={output}")
    return 0


def _require_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    required: list[tuple[str, str]],
) -> None:
    missing = [flag for attribute, flag in required if getattr(args, attribute) in (None, [])]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
