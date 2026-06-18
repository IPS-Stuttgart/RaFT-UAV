"""CLI for MMUAD sequence-level UAV type baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.classification import (
    classify_sequences_from_features,
    load_sequence_class_labels,
    sequence_features_from_files,
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
        required=True,
        help="CSV/JSON table with training sequence rows; may be repeated",
    )
    parser.add_argument(
        "--predict-feature-table",
        type=Path,
        action="append",
        required=True,
        help="CSV/JSON table with target sequence rows; may be repeated",
    )
    parser.add_argument(
        "--train-labels",
        type=Path,
        required=True,
        help="sequence class-map CSV/JSON/YAML or official Track 5 truth CSV/ZIP",
    )
    parser.add_argument(
        "--eval-labels",
        type=Path,
        help="optional class-map/truth labels for reporting target accuracy",
    )
    parser.add_argument(
        "--method",
        choices=(
            "majority",
            "nearest-neighbor",
            "nearest-centroid",
            "logistic-regression",
            "random-forest",
            "hist-gradient-boosting",
        ),
        default="nearest-neighbor",
    )
    parser.add_argument("--k", type=int, default=1, help="nearest-neighbor vote count")
    parser.add_argument("--output-class-map", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--train-features-csv", type=Path)
    parser.add_argument("--predict-features-csv", type=Path)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args(argv)

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
