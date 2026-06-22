"""Train/apply MMUAD sequence classifiers with image and fused evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.classification import (  # noqa: E402
    OFFICIAL_SEQUENCE_CLASS_LABELS,
    SEQUENCE_CLASSIFIER_METHODS,
    load_sequence_class_labels,
    save_sequence_classifier_model,
    sequence_classification_metrics,
    sequence_features_from_files,
    train_sequence_classifier_model,
)
from raft_uav.mmuad.image_evidence import (  # noqa: E402
    IMAGE_FEATURE_BACKENDS,
    build_image_evidence,
    write_image_evidence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "train MMUAD sequence-level image/non-image classifiers on train labels, "
            "apply to validation sequences, and write per-class probability/fusion artifacts"
        )
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--eval-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--non-image-train-feature-table",
        type=Path,
        action="append",
        default=[],
        help="candidate/sequence feature table for non-image train evidence; may be repeated",
    )
    parser.add_argument(
        "--non-image-val-feature-table",
        type=Path,
        action="append",
        default=[],
        help="candidate/sequence feature table for non-image validation evidence; may be repeated",
    )
    parser.add_argument("--train-sequence-glob", default="*")
    parser.add_argument("--val-sequence-glob", default="*")
    parser.add_argument("--train-timestamps-reference", type=Path)
    parser.add_argument("--val-timestamps-reference", type=Path)
    parser.add_argument("--timestamp-source", default="image")
    parser.add_argument("--max-frames-per-sequence", type=int, default=16)
    parser.add_argument("--max-image-time-delta-s", type=float, default=0.5)
    parser.add_argument(
        "--image-feature-backend",
        choices=IMAGE_FEATURE_BACKENDS,
        default="handcrafted",
    )
    parser.add_argument("--method", choices=SEQUENCE_CLASSIFIER_METHODS, default="random-forest")
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--fusion-weight-image", type=float, default=0.5)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_labels = load_sequence_class_labels(args.train_labels)
    eval_labels = load_sequence_class_labels(args.eval_labels) if args.eval_labels else None

    image_train, image_val = _build_image_feature_tables(args, output_dir)
    image_result = _train_probability_result(
        train_features=image_train,
        predict_features=image_val,
        train_labels=train_labels,
        eval_labels=eval_labels,
        method=args.method,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        model_name="image",
        output_dir=output_dir,
    )
    probability_results = {"image": image_result}

    if args.non_image_train_feature_table and args.non_image_val_feature_table:
        nonimage_train = sequence_features_from_files(args.non_image_train_feature_table)
        nonimage_val = sequence_features_from_files(args.non_image_val_feature_table)
        nonimage_train.to_csv(output_dir / "mmuad_nonimage_sequence_features_train.csv", index=False)
        nonimage_val.to_csv(output_dir / "mmuad_nonimage_sequence_features_val.csv", index=False)
        nonimage_result = _train_probability_result(
            train_features=nonimage_train,
            predict_features=nonimage_val,
            train_labels=train_labels,
            eval_labels=eval_labels,
            method=args.method,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            model_name="nonimage",
            output_dir=output_dir,
        )
        probability_results["nonimage"] = nonimage_result
        fused = _fuse_probabilities(
            image_result.probabilities,
            nonimage_result.probabilities,
            image_weight=args.fusion_weight_image,
            eval_labels=eval_labels,
        )
        fused_csv = output_dir / "mmuad_fused_classifier_probabilities.csv"
        fused.to_csv(fused_csv, index=False)
        fused_metrics = sequence_classification_metrics(
            fused[["sequence_id", "predicted_class"]],
            eval_labels=eval_labels,
        )
        probability_results["fused"] = ProbabilityResult(
            probabilities=fused,
            metrics=fused_metrics,
            model_path=None,
            feature_columns=[],
            csv_path=fused_csv,
        )

    summary = _summary(probability_results, args)
    summary_json = output_dir / "mmuad_image_sequence_classifier_summary.json"
    summary_csv = output_dir / "mmuad_image_sequence_classifier_summary.csv"
    summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    pd.DataFrame.from_records(summary["rows"]).to_csv(summary_csv, index=False)

    print("mmuad_image_sequence_classifier_train_val=ok")
    print(f"output_dir={output_dir}")
    print(f"summary_json={summary_json}")
    for row in summary["rows"]:
        accuracy = row.get("sequence_accuracy")
        print(f"{row['model']}_sequence_accuracy={accuracy}")
    return 0


class ProbabilityResult:
    def __init__(
        self,
        *,
        probabilities: pd.DataFrame,
        metrics: dict[str, Any],
        model_path: Path | None,
        feature_columns: list[str],
        csv_path: Path,
    ) -> None:
        self.probabilities = probabilities
        self.metrics = metrics
        self.model_path = model_path
        self.feature_columns = feature_columns
        self.csv_path = csv_path


def _build_image_feature_tables(args: argparse.Namespace, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_result = build_image_evidence(
        args.train_root,
        truth_file=args.train_timestamps_reference,
        sequence_glob=args.train_sequence_glob,
        timestamp_source=args.timestamp_source,
        max_frames_per_sequence=args.max_frames_per_sequence,
        max_image_time_delta_s=args.max_image_time_delta_s,
        image_feature_backend=args.image_feature_backend,
    )
    val_result = build_image_evidence(
        args.val_root,
        truth_file=args.val_timestamps_reference,
        sequence_glob=args.val_sequence_glob,
        timestamp_source=args.timestamp_source,
        max_frames_per_sequence=args.max_frames_per_sequence,
        max_image_time_delta_s=args.max_image_time_delta_s,
        image_feature_backend=args.image_feature_backend,
    )
    train_paths = write_image_evidence(train_result, output_dir / "image_train")
    val_paths = write_image_evidence(val_result, output_dir / "image_val")
    train_features = train_result.sequence_features.copy()
    val_features = val_result.sequence_features.copy()
    train_features.to_csv(output_dir / "mmuad_image_sequence_features_train.csv", index=False)
    val_features.to_csv(output_dir / "mmuad_image_sequence_features_val.csv", index=False)
    (output_dir / "mmuad_image_evidence_paths.json").write_text(
        json.dumps({"train": train_paths, "val": val_paths}, indent=2),
        encoding="utf-8",
    )
    return train_features, val_features


def _train_probability_result(
    *,
    train_features: pd.DataFrame,
    predict_features: pd.DataFrame,
    train_labels: dict[str, str],
    eval_labels: dict[str, str] | None,
    method: str,
    random_state: int,
    n_estimators: int,
    max_depth: int | None,
    model_name: str,
    output_dir: Path,
) -> ProbabilityResult:
    training = train_sequence_classifier_model(
        train_features=train_features,
        train_labels=train_labels,
        method=method,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    model_path = output_dir / f"mmuad_{model_name}_sequence_classifier.joblib"
    save_sequence_classifier_model(training.model, model_path)
    probabilities = _predict_probabilities(training.model, predict_features)
    probabilities = _with_prediction_columns(
        probabilities,
        class_source=f"sequence-{model_name}-{method}",
        eval_labels=eval_labels,
    )
    csv_path = output_dir / f"mmuad_{model_name}_classifier_probabilities.csv"
    probabilities.to_csv(csv_path, index=False)
    metrics = sequence_classification_metrics(
        probabilities[["sequence_id", "predicted_class"]],
        eval_labels=eval_labels,
    )
    metrics.update(
        {
            "method": method,
            "feature_count": len(training.model.get("feature_columns", [])),
            "train_sequence_count": len(training.model.get("train_sequences", [])),
        }
    )
    return ProbabilityResult(
        probabilities=probabilities,
        metrics=metrics,
        model_path=model_path,
        feature_columns=[str(value) for value in training.model.get("feature_columns", [])],
        csv_path=csv_path,
    )


def _predict_probabilities(model: dict[str, Any], predict_features: pd.DataFrame) -> pd.DataFrame:
    if predict_features.empty:
        raise ValueError("no prediction feature rows were provided")
    features = predict_features.copy()
    features["sequence_id"] = features["sequence_id"].astype(str)
    feature_columns = [str(column) for column in model.get("feature_columns", [])]
    if not feature_columns:
        raise ValueError("sequence classifier model has no feature_columns")
    matrix = _transform_model_feature_matrix(
        features,
        feature_columns,
        np.asarray(model.get("feature_means", []), dtype=float),
        np.asarray(model.get("feature_scales", []), dtype=float),
    )
    method = str(model.get("method", "")).strip().lower()
    if method == "majority":
        label = str(model.get("majority_class", "0"))
        return _one_hot_probability_frame(features["sequence_id"], [label] * len(features))
    if method == "nearest-neighbor":
        return _nearest_neighbor_probability_frame(model, features, matrix)
    if method == "nearest-centroid":
        return _nearest_centroid_probability_frame(model, features, matrix)
    return _sklearn_probability_frame(model, features, matrix)


def _transform_model_feature_matrix(
    rows: pd.DataFrame,
    columns: list[str],
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if means.shape[0] != len(columns) or scales.shape[0] != len(columns):
        raise ValueError("sequence classifier model feature statistics do not match feature_columns")
    matrix = np.column_stack(
        [
            pd.to_numeric(
                rows[column] if column in rows.columns else pd.Series(np.nan, index=rows.index),
                errors="coerce",
            ).to_numpy(float)
            for column in columns
        ]
    )
    matrix = np.where(np.isfinite(matrix), matrix, means)
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-9), scales, 1.0)
    return (matrix - means) / scales


def _one_hot_probability_frame(sequence_ids: pd.Series, labels: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sequence_id, label in zip(sequence_ids.astype(str), labels, strict=False):
        weights = {class_label: float(str(label) == class_label) for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS}
        records.append(_probability_record(sequence_id, weights))
    return pd.DataFrame.from_records(records)


def _nearest_neighbor_probability_frame(
    model: dict[str, Any],
    features: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    train_matrix = np.asarray(model.get("train_matrix", []), dtype=float)
    labels = [str(label) for label in model.get("train_labels", [])]
    if train_matrix.ndim != 2 or train_matrix.shape[0] == 0:
        raise ValueError("nearest-neighbor sequence classifier model has no train_matrix")
    records: list[dict[str, Any]] = []
    for row_idx, sequence_id in enumerate(features["sequence_id"].astype(str)):
        distances = np.linalg.norm(train_matrix - matrix[row_idx], axis=1)
        weights: dict[str, float] = {}
        for label, distance in zip(labels, distances, strict=False):
            weights[label] = weights.get(label, 0.0) + 1.0 / max(float(distance), 1.0e-9)
        records.append(_probability_record(sequence_id, weights))
    return pd.DataFrame.from_records(records)


def _nearest_centroid_probability_frame(
    model: dict[str, Any],
    features: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    train_matrix = np.asarray(model.get("train_matrix", []), dtype=float)
    labels = np.asarray([str(label) for label in model.get("train_labels", [])])
    if train_matrix.ndim != 2 or train_matrix.shape[0] == 0:
        raise ValueError("nearest-centroid sequence classifier model has no train_matrix")
    centroids = [
        (label, train_matrix[labels == label].mean(axis=0))
        for label in sorted(set(labels.astype(str)))
    ]
    records: list[dict[str, Any]] = []
    for row_idx, sequence_id in enumerate(features["sequence_id"].astype(str)):
        weights = {
            label: 1.0 / max(float(np.linalg.norm(matrix[row_idx] - centroid)), 1.0e-9)
            for label, centroid in centroids
        }
        records.append(_probability_record(sequence_id, weights))
    return pd.DataFrame.from_records(records)


def _sklearn_probability_frame(
    model: dict[str, Any],
    features: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    estimator = model.get("estimator")
    if estimator is None:
        raise ValueError("sequence classifier model has no estimator")
    if not hasattr(estimator, "predict_proba"):
        labels = [str(label) for label in estimator.predict(matrix)]
        return _one_hot_probability_frame(features["sequence_id"], labels)
    probabilities = estimator.predict_proba(matrix)
    estimator_classes = [str(label) for label in getattr(estimator, "classes_", [])]
    records: list[dict[str, Any]] = []
    for row_idx, sequence_id in enumerate(features["sequence_id"].astype(str)):
        weights = {
            estimator_classes[column_idx]: float(probabilities[row_idx, column_idx])
            for column_idx in range(min(len(estimator_classes), probabilities.shape[1]))
        }
        records.append(_probability_record(sequence_id, weights))
    return pd.DataFrame.from_records(records)


def _probability_record(sequence_id: str, weights: dict[str, float]) -> dict[str, Any]:
    total = float(sum(value for value in weights.values() if np.isfinite(value) and value >= 0.0))
    record: dict[str, Any] = {"sequence_id": str(sequence_id)}
    for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS:
        value = float(weights.get(str(class_label), 0.0))
        record[f"predicted_probability_{class_label}"] = value / total if total > 0.0 else 0.0
    return record


def _with_prediction_columns(
    probabilities: pd.DataFrame,
    *,
    class_source: str,
    eval_labels: dict[str, str] | None,
) -> pd.DataFrame:
    out = probabilities.copy()
    prob_columns = _probability_columns(out)
    out["predicted_class"] = [
        str(column.removeprefix("predicted_probability_"))
        for column in out[prob_columns].idxmax(axis=1)
    ]
    out["class_source"] = class_source
    if eval_labels:
        label_map = {str(key): str(value) for key, value in eval_labels.items()}
        out["ground_truth_class"] = out["sequence_id"].astype(str).map(label_map)
        out["correct"] = out["predicted_class"].astype(str) == out["ground_truth_class"].astype(str)
    return out


def _fuse_probabilities(
    image_probs: pd.DataFrame,
    nonimage_probs: pd.DataFrame,
    *,
    image_weight: float,
    eval_labels: dict[str, str] | None,
) -> pd.DataFrame:
    image_weight = float(np.clip(image_weight, 0.0, 1.0))
    nonimage_weight = 1.0 - image_weight
    left = _probability_index(image_probs)
    right = _probability_index(nonimage_probs)
    left_ids = set(left.index.astype(str))
    right_ids = set(right.index.astype(str))
    sequence_ids = sorted(left_ids.union(right_ids))
    records: list[dict[str, Any]] = []
    for sequence_id in sequence_ids:
        record: dict[str, Any] = {"sequence_id": sequence_id}
        total = 0.0
        for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS:
            column = f"predicted_probability_{class_label}"
            image_value = _probability_value(left, sequence_id, column)
            nonimage_value = _probability_value(right, sequence_id, column)
            if sequence_id in left_ids and sequence_id in right_ids:
                value = image_weight * image_value + nonimage_weight * nonimage_value
            elif sequence_id in left_ids:
                value = image_value
            else:
                value = nonimage_value
            record[column] = float(value)
            total += float(value)
        if total > 0.0:
            for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS:
                column = f"predicted_probability_{class_label}"
                record[column] = float(record[column]) / total
        records.append(record)
    return _with_prediction_columns(
        pd.DataFrame.from_records(records),
        class_source=f"sequence-fused-image-weight-{image_weight:g}",
        eval_labels=eval_labels,
    )


def _probability_index(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    return out.set_index("sequence_id", drop=False)


def _probability_value(rows: pd.DataFrame, sequence_id: str, column: str) -> float:
    if sequence_id not in rows.index or column not in rows.columns:
        return 0.0
    value = pd.to_numeric(pd.Series([rows.loc[sequence_id, column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(float(value)) else 0.0


def _probability_columns(rows: pd.DataFrame) -> list[str]:
    columns = [
        f"predicted_probability_{class_label}"
        for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS
        if f"predicted_probability_{class_label}" in rows.columns
    ]
    if not columns:
        raise ValueError("probability table has no predicted_probability_* columns")
    return columns


def _summary(results: dict[str, ProbabilityResult], args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model_name, result in results.items():
        row = {
            "model": model_name,
            "probabilities_csv": str(result.csv_path),
            "model_path": str(result.model_path) if result.model_path is not None else "",
            "feature_count": len(result.feature_columns),
        }
        row.update(result.metrics)
        rows.append(row)
    return {
        "rows": rows,
        "provenance": {
            "train_root": str(args.train_root),
            "val_root": str(args.val_root),
            "train_labels": str(args.train_labels),
            "eval_labels": str(args.eval_labels) if args.eval_labels else None,
            "image_feature_backend": args.image_feature_backend,
            "max_frames_per_sequence": int(args.max_frames_per_sequence),
            "fusion_weight_image": float(args.fusion_weight_image),
            "method": args.method,
            "non_image_train_feature_tables": [
                str(path) for path in args.non_image_train_feature_table
            ],
            "non_image_val_feature_tables": [
                str(path) for path in args.non_image_val_feature_table
            ],
            "classification_prediction_mode": "sequence_level",
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
