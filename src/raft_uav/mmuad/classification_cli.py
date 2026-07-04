"""CLI for MMUAD sequence-level UAV type baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.classification import (
    OFFICIAL_SEQUENCE_CLASS_LABELS,
    SEQUENCE_CLASSIFIER_LOSO_PREDICTION_COLUMNS,
    SEQUENCE_CLASSIFIER_METHODS,
    _normalize_sequence_classifier_method,
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
        default="nearest-neighbor",
        help=(
            "classifier method or alias; canonical methods: "
            f"{', '.join(SEQUENCE_CLASSIFIER_METHODS)}"
        ),
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
    try:
        args.method = _normalize_sequence_classifier_method(args.method)
    except ValueError as exc:
        parser.error(str(exc))

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
    eval_labels = (
        load_sequence_class_labels(args.eval_labels) if args.eval_labels is not None else None
    )
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
    predictions = _build_loso_predictions(
        features=features,
        labels=labels,
        method=args.method,
        k=args.k,
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


def _build_loso_predictions(
    *,
    features: pd.DataFrame,
    labels: dict[str, str],
    method: str,
    k: int,
) -> pd.DataFrame:
    """Build LOSO predictions while honoring nearest-neighbor CLI options."""

    method = _normalize_sequence_classifier_method(method)
    k = max(1, int(k))
    if method != "nearest-neighbor" or k == 1:
        return build_sequence_classifier_loso_predictions(
            features=features,
            labels=labels,
            method=method,
        )
    return _build_k_nearest_neighbor_loso_predictions(features=features, labels=labels, k=k)


def _build_k_nearest_neighbor_loso_predictions(
    *,
    features: pd.DataFrame,
    labels: dict[str, str],
    k: int,
) -> pd.DataFrame:
    """Return LOSO nearest-neighbor predictions for the requested vote count."""

    if features.empty:
        raise ValueError("no sequence feature rows were provided")
    if "sequence_id" not in features.columns:
        raise ValueError("sequence features must contain a sequence_id column")

    feature_rows = features.copy()
    feature_rows["sequence_id"] = feature_rows["sequence_id"].astype(str)
    label_map = {str(key): str(value) for key, value in labels.items()}
    heldout_sequences = sorted(
        set(feature_rows["sequence_id"].astype(str)).intersection(label_map)
    )
    if len(heldout_sequences) < 2:
        raise ValueError("LOSO sequence classification needs at least two labeled sequences")

    records: list[dict[str, object]] = []
    for heldout_sequence in heldout_sequences:
        train_sequences = [sequence for sequence in heldout_sequences if sequence != heldout_sequence]
        fold_train_features = feature_rows.loc[
            feature_rows["sequence_id"].isin(train_sequences)
        ].reset_index(drop=True)
        fold_predict_features = feature_rows.loc[
            feature_rows["sequence_id"].eq(heldout_sequence)
        ].reset_index(drop=True)
        fold_labels = {sequence: label_map[sequence] for sequence in train_sequences}
        fold_result = classify_sequences_from_features(
            train_features=fold_train_features,
            train_labels=fold_labels,
            predict_features=fold_predict_features,
            method="nearest-neighbor",
            k=k,
        )
        prediction = fold_result.predictions.iloc[0]
        truth_class = str(label_map[heldout_sequence])
        predicted_class = str(prediction["predicted_class"])
        record: dict[str, object] = {
            "sequence": heldout_sequence,
            "heldout_sequence": heldout_sequence,
            "method": "nearest-neighbor",
            "truth_class": truth_class,
            "predicted_class": predicted_class,
            "correct": bool(predicted_class == truth_class),
            "train_sequences": ";".join(train_sequences),
            "feature_columns": ";".join(
                str(value) for value in fold_result.metrics.get("feature_columns", [])
            ),
        }
        for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS:
            column = f"predicted_probability_{class_label}"
            record[column] = float(predicted_class == str(class_label))
        records.append(record)

    return pd.DataFrame.from_records(records, columns=SEQUENCE_CLASSIFIER_LOSO_PREDICTION_COLUMNS)


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
