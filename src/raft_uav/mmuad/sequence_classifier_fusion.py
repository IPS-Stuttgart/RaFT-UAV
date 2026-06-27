"""Train-safe fusion selection for MMUAD sequence classifiers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.classification import (
    OFFICIAL_SEQUENCE_CLASS_LABELS,
    SEQUENCE_CLASSIFIER_METHODS,
    load_sequence_class_labels,
    predict_sequence_class_probabilities_from_model,
    save_sequence_classifier_model,
    train_sequence_classifier_model,
)


CV_SUMMARY_CSV = "mmuad_train_safe_fusion_weight_cv_summary.csv"
OOF_PREDICTIONS_CSV = "mmuad_train_safe_fusion_weight_oof_predictions.csv"
PREDICT_DIAGNOSTIC_CSV = "mmuad_train_safe_fusion_weight_predict_diagnostic.csv"
SELECTED_PROBABILITIES_CSV = "mmuad_train_selected_fused_classifier_probabilities.csv"
SELECTED_IMAGE_MODEL = "mmuad_train_selected_image_sequence_classifier.joblib"
SELECTED_NONIMAGE_MODEL = "mmuad_train_selected_nonimage_sequence_classifier.joblib"
MANIFEST_JSON = "mmuad_train_safe_fusion_weight_probe.json"


@dataclass(frozen=True)
class FusionModelSpec:
    """One classifier specification for train-CV fusion selection."""

    method: str
    n_estimators: int
    max_depth: int | None
    random_state: int

    @property
    def name(self) -> str:
        depth = "none" if self.max_depth is None else str(self.max_depth)
        return f"{self.method}_n{self.n_estimators}_depth{depth}"


@dataclass(frozen=True)
class FusionSelectionResult:
    """Artifacts from train-safe classifier fusion selection."""

    cv_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    selected_probabilities: pd.DataFrame
    predict_diagnostic: pd.DataFrame
    manifest: dict[str, Any]


def select_train_safe_fusion(
    *,
    image_train_features: pd.DataFrame,
    nonimage_train_features: pd.DataFrame,
    image_predict_features: pd.DataFrame,
    nonimage_predict_features: pd.DataFrame,
    train_labels: dict[str, str],
    eval_labels: dict[str, str] | None = None,
    model_specs: list[FusionModelSpec],
    image_weights: list[float],
    cv_folds: int = 5,
    cv_random_state: int = 20260627,
    output_dir: Path | None = None,
) -> FusionSelectionResult:
    """Select image/non-image fusion weight by training-label CV, then predict.

    The public/evaluation labels are used only for the final diagnostic table.
    They never affect model or fusion-weight selection.
    """

    if not model_specs:
        raise ValueError("at least one model specification is required")
    image_weights = [float(value) for value in image_weights]
    if not image_weights:
        raise ValueError("at least one image fusion weight is required")
    train_labels = {str(key): str(value) for key, value in train_labels.items()}
    image_train = _sequence_indexed(image_train_features, "image_train_features")
    nonimage_train = _sequence_indexed(nonimage_train_features, "nonimage_train_features")
    train_sequences = sorted(set(image_train.index).intersection(nonimage_train.index, train_labels))
    if len(train_sequences) < 2:
        raise ValueError("fusion selection needs at least two labeled train sequences")
    image_train = image_train.loc[train_sequences].reset_index()
    nonimage_train = nonimage_train.loc[train_sequences].reset_index()
    labels = np.asarray([train_labels[sequence] for sequence in train_sequences], dtype=str)
    split_count = _effective_cv_folds(labels, cv_folds)

    splits = _stratified_cv_splits(labels, split_count, int(cv_random_state))
    cv_rows: list[dict[str, Any]] = []
    oof_frames: list[pd.DataFrame] = []
    for spec in model_specs:
        fold_probability_pairs: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, str], int]] = []
        for fold, (train_idx, holdout_idx) in enumerate(splits):
            fold_labels = {
                train_sequences[index]: train_labels[train_sequences[index]]
                for index in train_idx
            }
            image_model = train_sequence_classifier_model(
                train_features=image_train.iloc[train_idx].reset_index(drop=True),
                train_labels=fold_labels,
                method=spec.method,
                random_state=spec.random_state,
                n_estimators=spec.n_estimators,
                max_depth=spec.max_depth,
            ).model
            nonimage_model = train_sequence_classifier_model(
                train_features=nonimage_train.iloc[train_idx].reset_index(drop=True),
                train_labels=fold_labels,
                method=spec.method,
                random_state=spec.random_state,
                n_estimators=spec.n_estimators,
                max_depth=spec.max_depth,
            ).model
            image_probs = predict_sequence_class_probabilities_from_model(
                image_model,
                image_train.iloc[holdout_idx].reset_index(drop=True),
            )
            nonimage_probs = predict_sequence_class_probabilities_from_model(
                nonimage_model,
                nonimage_train.iloc[holdout_idx].reset_index(drop=True),
            )
            holdout_labels = {
                train_sequences[index]: train_labels[train_sequences[index]]
                for index in holdout_idx
            }
            fold_probability_pairs.append((image_probs, nonimage_probs, holdout_labels, fold))
        for weight in image_weights:
            fold_frames: list[pd.DataFrame] = []
            for image_probs, nonimage_probs, holdout_labels, fold in fold_probability_pairs:
                fused = fuse_sequence_probabilities(
                    image_probs,
                    nonimage_probs,
                    image_weight=weight,
                    eval_labels=holdout_labels,
                    class_source=f"train-cv-fused-{spec.name}-image-weight-{weight:g}",
                )
                fused["fold"] = int(fold)
                fused["model_name"] = spec.name
                fused["method"] = spec.method
                fused["max_depth"] = "" if spec.max_depth is None else int(spec.max_depth)
                fused["n_estimators"] = int(spec.n_estimators)
                fused["image_weight"] = float(weight)
                fold_frames.append(fused)
            oof = pd.concat(fold_frames, ignore_index=True, sort=False)
            accuracy, correct = _accuracy_from_probability_rows(oof)
            cv_rows.append(
                {
                    "model_name": spec.name,
                    "method": spec.method,
                    "max_depth": "" if spec.max_depth is None else int(spec.max_depth),
                    "n_estimators": int(spec.n_estimators),
                    "image_weight": float(weight),
                    "train_cv_accuracy": accuracy,
                    "correct": correct,
                    "sequence_count": int(oof["sequence_id"].nunique()),
                }
            )
            oof_frames.append(oof)
    cv_summary = pd.DataFrame.from_records(cv_rows)
    cv_summary["_max_depth_sort"] = [
        _max_depth_sort_value(value) for value in cv_summary["max_depth"]
    ]
    cv_summary = cv_summary.sort_values(
        ["train_cv_accuracy", "_max_depth_sort", "image_weight", "model_name"],
        ascending=[False, True, True, True],
    ).drop(columns=["_max_depth_sort"]).reset_index(drop=True)
    selected = cv_summary.iloc[0].to_dict()
    selected_spec = _matching_spec(model_specs, selected)
    selected_weight = float(selected["image_weight"])
    image_model = train_sequence_classifier_model(
        train_features=image_train,
        train_labels=train_labels,
        method=selected_spec.method,
        random_state=selected_spec.random_state,
        n_estimators=selected_spec.n_estimators,
        max_depth=selected_spec.max_depth,
    ).model
    nonimage_model = train_sequence_classifier_model(
        train_features=nonimage_train,
        train_labels=train_labels,
        method=selected_spec.method,
        random_state=selected_spec.random_state,
        n_estimators=selected_spec.n_estimators,
        max_depth=selected_spec.max_depth,
    ).model
    image_predict = _sequence_indexed(image_predict_features, "image_predict_features").reset_index()
    nonimage_predict = _sequence_indexed(nonimage_predict_features, "nonimage_predict_features").reset_index()
    image_probs = predict_sequence_class_probabilities_from_model(image_model, image_predict)
    nonimage_probs = predict_sequence_class_probabilities_from_model(nonimage_model, nonimage_predict)
    selected_probabilities = fuse_sequence_probabilities(
        image_probs,
        nonimage_probs,
        image_weight=selected_weight,
        eval_labels=eval_labels,
        class_source=f"train-selected-fused-{selected_spec.name}-image-weight-{selected_weight:g}",
    )
    predict_rows: list[dict[str, Any]] = []
    for weight in image_weights:
        fused = fuse_sequence_probabilities(
            image_probs,
            nonimage_probs,
            image_weight=weight,
            eval_labels=eval_labels,
            class_source=f"diagnostic-fused-{selected_spec.name}-image-weight-{weight:g}",
        )
        accuracy, correct = _accuracy_from_probability_rows(fused)
        predict_rows.append(
            {
                "model_name": selected_spec.name,
                "image_weight": float(weight),
                "predict_accuracy_diagnostic": accuracy,
                "correct": correct,
                "sequence_count": int(len(fused)),
            }
        )
    predict_diagnostic = pd.DataFrame.from_records(predict_rows).sort_values(
        ["predict_accuracy_diagnostic", "image_weight"],
        ascending=[False, True],
    ).reset_index(drop=True)
    manifest = {
        "schema": "raft-uav-mmuad-train-safe-fusion-weight-v1",
        "selection_protocol": (
            "Stratified train-label CV selects model/fusion weight; "
            "evaluation labels are diagnostic only after selection."
        ),
        "train_sequence_count": int(len(train_sequences)),
        "cv_folds": int(split_count),
        "cv_random_state": int(cv_random_state),
        "class_labels": list(OFFICIAL_SEQUENCE_CLASS_LABELS),
        "image_weights": image_weights,
        "model_specs": [_spec_payload(spec) for spec in model_specs],
        "selected": _jsonable(selected),
        "selected_predict_accuracy_diagnostic": _accuracy_from_probability_rows(
            selected_probabilities
        )[0],
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_model_path = save_sequence_classifier_model(image_model, output_dir / SELECTED_IMAGE_MODEL)
        nonimage_model_path = save_sequence_classifier_model(
            nonimage_model,
            output_dir / SELECTED_NONIMAGE_MODEL,
        )
        manifest.update(
            {
                "selected_image_model_path": str(image_model_path),
                "selected_nonimage_model_path": str(nonimage_model_path),
                "cv_summary_csv": str(output_dir / CV_SUMMARY_CSV),
                "oof_predictions_csv": str(output_dir / OOF_PREDICTIONS_CSV),
                "predict_diagnostic_csv": str(output_dir / PREDICT_DIAGNOSTIC_CSV),
                "selected_probabilities_csv": str(output_dir / SELECTED_PROBABILITIES_CSV),
            }
        )
        cv_summary.to_csv(output_dir / CV_SUMMARY_CSV, index=False)
        pd.concat(oof_frames, ignore_index=True, sort=False).to_csv(
            output_dir / OOF_PREDICTIONS_CSV,
            index=False,
        )
        predict_diagnostic.to_csv(output_dir / PREDICT_DIAGNOSTIC_CSV, index=False)
        selected_probabilities.to_csv(output_dir / SELECTED_PROBABILITIES_CSV, index=False)
        (output_dir / MANIFEST_JSON).write_text(
            json.dumps(_jsonable(manifest), indent=2),
            encoding="utf-8",
        )
    return FusionSelectionResult(
        cv_summary=cv_summary,
        oof_predictions=pd.concat(oof_frames, ignore_index=True, sort=False),
        selected_probabilities=selected_probabilities,
        predict_diagnostic=predict_diagnostic,
        manifest=manifest,
    )


def fuse_sequence_probabilities(
    image_probabilities: pd.DataFrame,
    nonimage_probabilities: pd.DataFrame,
    *,
    image_weight: float,
    eval_labels: dict[str, str] | None = None,
    class_source: str | None = None,
) -> pd.DataFrame:
    """Blend per-class image and non-image probabilities by sequence."""

    image_weight = float(np.clip(float(image_weight), 0.0, 1.0))
    nonimage_weight = 1.0 - image_weight
    left = _probability_index(image_probabilities)
    right = _probability_index(nonimage_probabilities)
    sequence_ids = sorted(set(left.index.astype(str)).union(right.index.astype(str)))
    records: list[dict[str, Any]] = []
    for sequence_id in sequence_ids:
        record: dict[str, Any] = {"sequence_id": sequence_id}
        total = 0.0
        for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS:
            column = f"predicted_probability_{class_label}"
            if sequence_id in left.index and sequence_id in right.index:
                value = (
                    image_weight * _probability_value(left, sequence_id, column)
                    + nonimage_weight * _probability_value(right, sequence_id, column)
                )
            elif sequence_id in left.index:
                value = _probability_value(left, sequence_id, column)
            else:
                value = _probability_value(right, sequence_id, column)
            record[column] = float(value)
            total += float(value)
        if total > 0.0:
            for class_label in OFFICIAL_SEQUENCE_CLASS_LABELS:
                column = f"predicted_probability_{class_label}"
                record[column] = float(record[column]) / total
        records.append(record)
    out = pd.DataFrame.from_records(records)
    probability_columns = [f"predicted_probability_{label}" for label in OFFICIAL_SEQUENCE_CLASS_LABELS]
    out["predicted_class"] = [
        str(column).removeprefix("predicted_probability_")
        for column in out[probability_columns].idxmax(axis=1)
    ]
    out["class_source"] = class_source or f"sequence-fused-image-weight-{image_weight:g}"
    if eval_labels:
        label_map = {str(key): str(value) for key, value in eval_labels.items()}
        out["ground_truth_class"] = out["sequence_id"].astype(str).map(label_map)
        out["correct"] = (
            out["predicted_class"].astype(str) == out["ground_truth_class"].astype(str)
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "select MMUAD image/non-image sequence-classifier fusion settings "
            "using train-label cross-validation, then apply the frozen setting"
        )
    )
    parser.add_argument("--image-train-features", type=Path, required=True)
    parser.add_argument("--nonimage-train-features", type=Path, required=True)
    parser.add_argument("--image-predict-features", type=Path, required=True)
    parser.add_argument("--nonimage-predict-features", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--eval-labels", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=SEQUENCE_CLASSIFIER_METHODS,
        action="append",
        default=None,
        help="classifier method to consider; may be repeated",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--max-depth-grid",
        default="none,8,4",
        help="comma list for tree depth candidates; use 'none' for unlimited",
    )
    parser.add_argument(
        "--image-weight-grid",
        default="0:1:0.05",
        help="comma list or start:stop:step grid for image fusion weights",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=20260627)
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args(argv)

    methods = args.method or ["random-forest"]
    depths = _parse_max_depth_grid(args.max_depth_grid)
    model_specs = [
        FusionModelSpec(
            method=method,
            n_estimators=int(args.n_estimators),
            max_depth=depth if method == "random-forest" else None,
            random_state=int(args.random_state),
        )
        for method in methods
        for depth in (depths if method == "random-forest" else [None])
    ]
    result = select_train_safe_fusion(
        image_train_features=pd.read_csv(args.image_train_features),
        nonimage_train_features=pd.read_csv(args.nonimage_train_features),
        image_predict_features=pd.read_csv(args.image_predict_features),
        nonimage_predict_features=pd.read_csv(args.nonimage_predict_features),
        train_labels=load_sequence_class_labels(args.train_labels),
        eval_labels=load_sequence_class_labels(args.eval_labels) if args.eval_labels else None,
        model_specs=model_specs,
        image_weights=_parse_weight_grid(args.image_weight_grid),
        cv_folds=int(args.cv_folds),
        cv_random_state=int(args.cv_random_state),
        output_dir=args.output_dir,
    )
    selected = result.manifest["selected"]
    print("mmuad_train_safe_fusion_weight=ok")
    print(f"output_dir={Path(args.output_dir)}")
    print(f"selected_model={selected['model_name']}")
    print(f"selected_image_weight={selected['image_weight']}")
    print(f"selected_train_cv_accuracy={selected['train_cv_accuracy']}")
    print(f"selected_probabilities_csv={Path(args.output_dir) / SELECTED_PROBABILITIES_CSV}")
    return 0


def _sequence_indexed(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    if rows.empty:
        raise ValueError(f"{name} is empty")
    if "sequence_id" not in rows.columns:
        raise ValueError(f"{name} must contain a sequence_id column")
    out = rows.copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    return out.drop_duplicates("sequence_id", keep="first").set_index("sequence_id", drop=True)


def _effective_cv_folds(labels: np.ndarray, cv_folds: int) -> int:
    counts = pd.Series(labels.astype(str)).value_counts()
    min_count = int(counts.min()) if not counts.empty else 0
    split_count = min(int(cv_folds), min_count)
    if split_count < 2:
        raise ValueError("stratified fusion selection needs at least two examples per class")
    return split_count


def _stratified_cv_splits(
    labels: np.ndarray,
    split_count: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=str)
    rng = np.random.default_rng(int(random_state))
    holdout_by_fold: list[list[int]] = [[] for _ in range(int(split_count))]
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label).astype(int)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for fold, fold_indices in enumerate(np.array_split(shuffled, int(split_count))):
            holdout_by_fold[fold].extend(int(index) for index in fold_indices)
    all_indices = np.arange(len(labels), dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for holdout in holdout_by_fold:
        holdout_idx = np.asarray(sorted(holdout), dtype=int)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[holdout_idx] = False
        splits.append((all_indices[train_mask], holdout_idx))
    return splits


def _accuracy_from_probability_rows(rows: pd.DataFrame) -> tuple[float | None, int | None]:
    if "ground_truth_class" not in rows.columns:
        return None, None
    scored = rows.loc[rows["ground_truth_class"].notna()].copy()
    if scored.empty:
        return None, None
    correct = scored["predicted_class"].astype(str) == scored["ground_truth_class"].astype(str)
    return float(correct.mean()), int(correct.sum())


def _matching_spec(model_specs: list[FusionModelSpec], selected: dict[str, Any]) -> FusionModelSpec:
    for spec in model_specs:
        if spec.name == str(selected["model_name"]):
            return spec
    raise ValueError(f"selected model spec {selected['model_name']!r} was not found")


def _spec_payload(spec: FusionModelSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "method": spec.method,
        "n_estimators": int(spec.n_estimators),
        "max_depth": spec.max_depth,
        "random_state": int(spec.random_state),
    }


def _parse_weight_grid(text: str) -> list[float]:
    text = str(text).strip()
    if ":" in text:
        start_s, stop_s, step_s = text.split(":", 2)
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step <= 0:
            raise ValueError("image weight grid step must be positive")
        values = []
        value = start
        while value <= stop + step / 10.0:
            values.append(round(float(value), 10))
            value += step
        return values
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_max_depth_grid(text: str) -> list[int | None]:
    values: list[int | None] = []
    for item in str(text).split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        values.append(None if stripped in {"none", "null", "unlimited"} else int(stripped))
    return values or [None]


def _max_depth_sort_value(value: Any) -> float:
    if value in {None, "", "none", "None"}:
        return float("inf")
    try:
        return float(value)
    except Exception:
        return float("inf")


def _probability_index(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    return out.set_index("sequence_id", drop=False)


def _probability_value(rows: pd.DataFrame, sequence_id: str, column: str) -> float:
    if sequence_id not in rows.index or column not in rows.columns:
        return 0.0
    value = pd.to_numeric(pd.Series([rows.loc[sequence_id, column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(float(value)) else 0.0


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
