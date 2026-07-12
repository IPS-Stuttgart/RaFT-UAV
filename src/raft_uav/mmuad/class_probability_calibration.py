"""Train-only calibration for MMUAD sequence class probabilities.

Classification probabilities are used by several MMUAD pose components, including
class-conditioned candidate uncertainty and anchor reliability.  Raw classifier
probabilities can be substantially overconfident even when sequence-level accuracy is
high.  This module fits one scalar temperature on out-of-fold training predictions and
applies the frozen calibration to validation or test probabilities without using pose
or class truth at inference time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from raft_uav.mmuad.classification import load_sequence_class_labels


MODEL_SCHEMA = "raft-uav-mmuad-class-probability-calibrator-v1"
_SEQUENCE_COLUMN_CANDIDATES = (
    "sequence_id",
    "sequence",
    "Sequence",
    "heldout_sequence",
)
_PROBABILITY_PREFIXES = (
    "class_prob_",
    "image_class_prob_",
    "predicted_probability_",
    "probability_",
)


@dataclass(frozen=True)
class ClassProbabilityCalibrator:
    """Serializable scalar-temperature calibrator."""

    schema: str
    method: str
    temperature: float
    class_labels: list[str]
    source_probability_columns: list[str]
    output_prefix: str = "calibrated_class_prob_"
    epsilon: float = 1.0e-9


def fit_temperature_calibrator(
    predictions: pd.DataFrame,
    labels: Mapping[str, str] | pd.DataFrame,
    *,
    probability_columns: Sequence[str] | None = None,
    sequence_column: str | None = None,
    label_column: str | None = None,
    output_prefix: str = "calibrated_class_prob_",
    epsilon: float = 1.0e-9,
    min_temperature: float = 0.05,
    max_temperature: float = 20.0,
    ece_bins: int = 10,
) -> tuple[ClassProbabilityCalibrator, dict[str, Any]]:
    """Fit a scalar temperature from out-of-fold sequence predictions.

    The caller is responsible for supplying training-only out-of-fold probabilities.
    Ground-truth labels are used only during this fit step.
    """

    rows = pd.DataFrame(predictions).copy()
    sequence_column = resolve_sequence_column(rows, sequence_column)
    columns, class_labels = resolve_probability_columns(rows, probability_columns)
    label_map = normalize_label_map(labels, label_column=label_column)

    sequence_ids = rows[sequence_column].astype(str)
    truth_labels = sequence_ids.map(label_map)
    probabilities = normalized_probabilities(rows, columns, epsilon=epsilon)
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    truth_indices = truth_labels.map(class_to_index)
    valid = truth_indices.notna() & np.isfinite(probabilities).all(axis=1)
    if not valid.any():
        raise ValueError("no prediction rows have both finite probabilities and class labels")

    fit_probabilities = probabilities[valid.to_numpy(bool)]
    fit_truth = truth_indices.loc[valid].astype(int).to_numpy()
    if len(np.unique(fit_truth)) < 2:
        raise ValueError("temperature calibration requires at least two observed classes")

    lower = max(float(min_temperature), 1.0e-4)
    upper = max(float(max_temperature), lower + 1.0e-4)

    def objective(temperature: float) -> float:
        calibrated = temperature_scale_probabilities(
            fit_probabilities,
            temperature=float(temperature),
            epsilon=epsilon,
        )
        return multiclass_nll(calibrated, fit_truth, epsilon=epsilon)

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    temperature = float(result.x) if result.success and np.isfinite(result.x) else 1.0
    calibrated = temperature_scale_probabilities(
        fit_probabilities,
        temperature=temperature,
        epsilon=epsilon,
    )
    before = classification_calibration_metrics(
        fit_probabilities,
        fit_truth,
        ece_bins=ece_bins,
        epsilon=epsilon,
    )
    after = classification_calibration_metrics(
        calibrated,
        fit_truth,
        ece_bins=ece_bins,
        epsilon=epsilon,
    )
    model = ClassProbabilityCalibrator(
        schema=MODEL_SCHEMA,
        method="temperature",
        temperature=temperature,
        class_labels=list(class_labels),
        source_probability_columns=list(columns),
        output_prefix=str(output_prefix),
        epsilon=float(epsilon),
    )
    summary = {
        "schema": MODEL_SCHEMA,
        "protocol": "fit on train-only out-of-fold class probabilities",
        "row_count": int(len(rows)),
        "matched_row_count": int(valid.sum()),
        "matched_sequence_count": int(sequence_ids.loc[valid].nunique()),
        "class_labels": list(class_labels),
        "probability_columns": list(columns),
        "temperature": temperature,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "before": before,
        "after": after,
        "nll_improvement": float(before["nll"] - after["nll"]),
        "brier_improvement": float(before["brier"] - after["brier"]),
        "ece_improvement": float(before["ece"] - after["ece"]),
    }
    return model, summary


def apply_temperature_calibrator(
    predictions: pd.DataFrame,
    model: ClassProbabilityCalibrator,
    *,
    probability_columns: Sequence[str] | None = None,
    output_prefix: str | None = None,
    replace_probabilities: bool = False,
) -> pd.DataFrame:
    """Apply a frozen calibrator without requiring labels."""

    if model.schema != MODEL_SCHEMA:
        raise ValueError(f"unsupported class-probability calibrator schema: {model.schema!r}")
    rows = pd.DataFrame(predictions).copy()
    explicit = probability_columns
    if explicit is None and all(column in rows.columns for column in model.source_probability_columns):
        explicit = model.source_probability_columns
    columns, labels = resolve_probability_columns(
        rows,
        explicit,
        class_labels=model.class_labels,
    )
    if list(labels) != list(model.class_labels):
        raise ValueError(
            "class labels in prediction probabilities do not match the fitted calibrator: "
            f"expected {model.class_labels}, got {labels}"
        )
    probabilities = normalized_probabilities(rows, columns, epsilon=model.epsilon)
    calibrated = temperature_scale_probabilities(
        probabilities,
        temperature=model.temperature,
        epsilon=model.epsilon,
    )
    prefix = model.output_prefix if output_prefix is None else str(output_prefix)
    for index, label in enumerate(model.class_labels):
        rows[f"{prefix}{label}"] = calibrated[:, index]
    rows["class_probability_temperature"] = float(model.temperature)
    rows["class_probability_calibrated"] = True
    rows["class_probability_entropy_raw"] = probability_entropy(
        probabilities,
        epsilon=model.epsilon,
    )
    rows["class_probability_entropy_calibrated"] = probability_entropy(
        calibrated,
        epsilon=model.epsilon,
    )
    if replace_probabilities:
        for index, column in enumerate(columns):
            raw_column = f"raw_{column}"
            if raw_column not in rows.columns:
                rows[raw_column] = rows[column]
            rows[column] = calibrated[:, index]
    return rows


def temperature_scale_probabilities(
    probabilities: np.ndarray,
    *,
    temperature: float,
    epsilon: float = 1.0e-9,
) -> np.ndarray:
    """Apply scalar temperature scaling to a probability matrix."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional multi-class matrix")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    values = np.clip(values, float(epsilon), 1.0)
    values = values / values.sum(axis=1, keepdims=True)
    logits = np.log(values) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def classification_calibration_metrics(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    ece_bins: int = 10,
    epsilon: float = 1.0e-9,
) -> dict[str, float]:
    """Return accuracy, NLL, Brier score, and expected calibration error."""

    values = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth_indices, dtype=int)
    predictions = np.argmax(values, axis=1)
    one_hot = np.eye(values.shape[1], dtype=float)[truth]
    return {
        "accuracy": float(np.mean(predictions == truth)),
        "nll": multiclass_nll(values, truth, epsilon=epsilon),
        "brier": float(np.mean(np.sum((values - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(values, truth, bins=ece_bins),
        "mean_confidence": float(np.mean(np.max(values, axis=1))),
    }


def multiclass_nll(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    epsilon: float = 1.0e-9,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth_indices, dtype=int)
    selected = values[np.arange(len(values)), truth]
    return float(-np.mean(np.log(np.clip(selected, float(epsilon), 1.0))))


def expected_calibration_error(
    probabilities: np.ndarray,
    truth_indices: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    truth = np.asarray(truth_indices, dtype=int)
    confidence = np.max(values, axis=1)
    correct = np.argmax(values, axis=1) == truth
    edges = np.linspace(0.0, 1.0, max(int(bins), 1) + 1)
    error = 0.0
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        if index == len(edges) - 2:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not mask.any():
            continue
        weight = float(np.mean(mask))
        error += weight * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
    return float(error)


def probability_entropy(probabilities: np.ndarray, *, epsilon: float = 1.0e-9) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), float(epsilon), 1.0)
    return -np.sum(values * np.log(values), axis=1)


def normalized_probabilities(
    rows: pd.DataFrame,
    columns: Sequence[str],
    *,
    epsilon: float = 1.0e-9,
) -> np.ndarray:
    values = rows[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(values).all(axis=1)
    positive_sum = np.nansum(np.maximum(values, 0.0), axis=1) > 0.0
    valid = finite & positive_sum
    normalized = np.full_like(values, np.nan, dtype=float)
    if valid.any():
        clipped = np.clip(values[valid], 0.0, None)
        clipped = np.clip(clipped, float(epsilon), None)
        normalized[valid] = clipped / clipped.sum(axis=1, keepdims=True)
    return normalized


def resolve_probability_columns(
    rows: pd.DataFrame,
    probability_columns: Sequence[str] | None = None,
    *,
    class_labels: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve a stable ordered probability-column set."""

    if probability_columns is not None:
        columns = [str(column) for column in probability_columns]
        missing = [column for column in columns if column not in rows.columns]
        if missing:
            raise ValueError(f"prediction table missing probability columns: {missing}")
        labels = [_probability_column_label(column) for column in columns]
    else:
        candidates: list[tuple[int, str, list[str], list[str]]] = []
        for prefix in _PROBABILITY_PREFIXES:
            matched = [str(column) for column in rows.columns if str(column).startswith(prefix)]
            labeled = [
                (column, str(column)[len(prefix) :])
                for column in matched
                if str(column)[len(prefix) :]
            ]
            labeled.sort(key=lambda item: _class_label_sort_key(item[1]))
            if len(labeled) >= 2:
                candidates.append(
                    (
                        len(labeled),
                        prefix,
                        [column for column, _ in labeled],
                        [label for _, label in labeled],
                    )
                )
        if not candidates:
            raise ValueError("could not find at least two class-probability columns")
        _count, _prefix, columns, labels = sorted(candidates, reverse=True)[0]
    if len(set(labels)) != len(labels):
        raise ValueError(f"probability columns have duplicate class labels: {labels}")
    if class_labels is not None:
        desired = [str(label) for label in class_labels]
        mapping = dict(zip(labels, columns, strict=True))
        missing = [label for label in desired if label not in mapping]
        if missing:
            raise ValueError(f"prediction table missing calibrated classes: {missing}")
        columns = [mapping[label] for label in desired]
        labels = desired
    return columns, labels


def resolve_sequence_column(rows: pd.DataFrame, sequence_column: str | None = None) -> str:
    if sequence_column is not None:
        if sequence_column not in rows.columns:
            raise ValueError(f"prediction table missing sequence column {sequence_column!r}")
        return str(sequence_column)
    for column in _SEQUENCE_COLUMN_CANDIDATES:
        if column in rows.columns:
            return column
    raise ValueError("could not resolve a sequence identifier column")


def normalize_label_map(
    labels: Mapping[str, str] | pd.DataFrame,
    *,
    label_column: str | None = None,
) -> dict[str, str]:
    if isinstance(labels, Mapping):
        return {str(sequence): str(label) for sequence, label in labels.items()}
    rows = pd.DataFrame(labels).copy()
    sequence_column = resolve_sequence_column(rows)
    if label_column is None:
        for candidate in ("truth_class", "uav_type", "Classification", "class_id"):
            if candidate in rows.columns:
                label_column = candidate
                break
    if label_column is None or label_column not in rows.columns:
        raise ValueError("could not resolve a class-label column")
    return {
        str(sequence): str(label)
        for sequence, label in zip(rows[sequence_column], rows[label_column], strict=False)
    }


def save_calibrator(model: ClassProbabilityCalibrator, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")
    return path


def load_calibrator(path: Path) -> ClassProbabilityCalibrator:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClassProbabilityCalibrator(
        schema=str(payload["schema"]),
        method=str(payload["method"]),
        temperature=float(payload["temperature"]),
        class_labels=[str(value) for value in payload["class_labels"]],
        source_probability_columns=[
            str(value) for value in payload["source_probability_columns"]
        ],
        output_prefix=str(payload.get("output_prefix", "calibrated_class_prob_")),
        epsilon=float(payload.get("epsilon", 1.0e-9)),
    )


def _probability_column_label(column: str) -> str:
    for prefix in _PROBABILITY_PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column.rsplit("_", 1)[-1]


def _class_label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, str(label))


def _read_csv_as_strings(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_json(payload: Mapping[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _add_probability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--probability-column", action="append", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--replace-probabilities", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="fit on train-only OOF probabilities")
    fit_parser.add_argument("--predictions-csv", type=Path, required=True)
    fit_parser.add_argument("--labels-csv", type=Path, required=True)
    fit_parser.add_argument("--model-json", type=Path, required=True)
    fit_parser.add_argument("--output-csv", type=Path)
    fit_parser.add_argument("--summary-json", type=Path)
    fit_parser.add_argument("--sequence-column")
    fit_parser.add_argument("--min-temperature", type=float, default=0.05)
    fit_parser.add_argument("--max-temperature", type=float, default=20.0)
    fit_parser.add_argument("--ece-bins", type=int, default=10)
    _add_probability_arguments(fit_parser)

    apply_parser = subparsers.add_parser("apply", help="apply a frozen calibrator")
    apply_parser.add_argument("--predictions-csv", type=Path, required=True)
    apply_parser.add_argument("--model-json", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)
    _add_probability_arguments(apply_parser)

    args = parser.parse_args(argv)
    predictions = _read_csv_as_strings(args.predictions_csv)
    if args.command == "fit":
        label_map = load_sequence_class_labels(args.labels_csv)
        prefix = args.output_prefix or "calibrated_class_prob_"
        model, summary = fit_temperature_calibrator(
            predictions,
            label_map,
            probability_columns=args.probability_column,
            sequence_column=args.sequence_column,
            output_prefix=prefix,
            min_temperature=args.min_temperature,
            max_temperature=args.max_temperature,
            ece_bins=args.ece_bins,
        )
        save_calibrator(model, args.model_json)
        _write_json(summary, args.summary_json)
        if args.output_csv is not None:
            calibrated = apply_temperature_calibrator(
                predictions,
                model,
                probability_columns=args.probability_column,
                output_prefix=prefix,
                replace_probabilities=args.replace_probabilities,
            )
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            calibrated.to_csv(args.output_csv, index=False)
        print(f"class_probability_calibrator={args.model_json}")
        print(f"temperature={model.temperature}")
        print(f"nll_improvement={summary['nll_improvement']}")
        return 0

    model = load_calibrator(args.model_json)
    calibrated = apply_temperature_calibrator(
        predictions,
        model,
        probability_columns=args.probability_column,
        output_prefix=args.output_prefix,
        replace_probabilities=args.replace_probabilities,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(args.output_csv, index=False)
    print(f"calibrated_predictions={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
