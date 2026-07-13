"""Train-safe entropy-adaptive fusion for MMUAD sequence classifiers.

A single global image/non-image fusion weight cannot express that the most
reliable modality may vary by sequence. This module keeps a train-selected global
prior, then shifts the effective weight toward the lower-entropy modality for
individual sequences. Setting ``adaptation_strength=0`` recovers ordinary global
linear fusion exactly, so cross-validation can reject adaptation when it does not
help.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


OFFICIAL_SEQUENCE_CLASS_LABELS = ("0", "1", "2", "3")
PROBABILITY_COLUMNS = tuple(
    f"predicted_probability_{label}" for label in OFFICIAL_SEQUENCE_CLASS_LABELS
)
CV_SUMMARY_CSV = "mmuad_entropy_adaptive_fusion_cv_summary.csv"
SELECTED_PROBABILITIES_CSV = "mmuad_entropy_adaptive_fusion_probabilities.csv"
MANIFEST_JSON = "mmuad_entropy_adaptive_fusion.json"
SELECTION_METRICS = ("accuracy", "balanced_accuracy", "log_loss")


@dataclass(frozen=True)
class EntropyAdaptiveFusionConfig:
    """Controls for sequence-specific entropy-adaptive probability fusion."""

    prior_image_weight: float = 0.5
    adaptation_strength: float = 0.5
    entropy_power: float = 1.0
    probability_floor: float = 1e-9
    min_image_weight: float = 0.0
    max_image_weight: float = 1.0


@dataclass(frozen=True)
class EntropyAdaptiveFusionSelectionResult:
    """Train-only grid-selection outputs and frozen prediction probabilities."""

    cv_summary: pd.DataFrame
    selected_probabilities: pd.DataFrame
    manifest: dict[str, Any]


def fuse_entropy_adaptive_probabilities(
    image_probabilities: pd.DataFrame,
    nonimage_probabilities: pd.DataFrame,
    *,
    config: EntropyAdaptiveFusionConfig | None = None,
    eval_labels: dict[str, str] | None = None,
    class_source: str | None = None,
) -> pd.DataFrame:
    """Fuse probability rows with a confidence-dependent per-sequence weight."""

    config = config or EntropyAdaptiveFusionConfig()
    _validate_config(config)
    image = _normalized_probability_index(image_probabilities, "image_probabilities")
    nonimage = _normalized_probability_index(
        nonimage_probabilities,
        "nonimage_probabilities",
    )
    sequence_ids = sorted(set(image.index).union(nonimage.index))
    records: list[dict[str, Any]] = []
    for sequence_id in sequence_ids:
        image_available = sequence_id in image.index
        nonimage_available = sequence_id in nonimage.index
        if not image_available and not nonimage_available:
            continue
        image_vector = _probability_vector(image, sequence_id) if image_available else None
        nonimage_vector = (
            _probability_vector(nonimage, sequence_id) if nonimage_available else None
        )
        diagnostics = _fusion_diagnostics(
            image_vector,
            nonimage_vector,
            config=config,
        )
        if image_vector is None:
            fused = np.asarray(nonimage_vector, dtype=float)
        elif nonimage_vector is None:
            fused = np.asarray(image_vector, dtype=float)
        else:
            weight = float(diagnostics["image_weight_effective"])
            fused = weight * image_vector + (1.0 - weight) * nonimage_vector
        fused = _normalize_vector(fused, probability_floor=config.probability_floor)
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "image_available": bool(image_available),
            "nonimage_available": bool(nonimage_available),
            **diagnostics,
        }
        for column, value in zip(PROBABILITY_COLUMNS, fused, strict=True):
            record[column] = float(value)
        record["predicted_class"] = OFFICIAL_SEQUENCE_CLASS_LABELS[int(np.argmax(fused))]
        records.append(record)
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return _empty_output()
    source = class_source or (
        "sequence-entropy-adaptive-fusion-"
        f"prior-{config.prior_image_weight:g}-strength-{config.adaptation_strength:g}-"
        f"power-{config.entropy_power:g}"
    )
    out["class_source"] = source
    if eval_labels:
        labels = {str(key): str(value) for key, value in eval_labels.items()}
        out["ground_truth_class"] = out["sequence_id"].astype(str).map(labels)
        out["correct"] = (
            out["predicted_class"].astype(str)
            == out["ground_truth_class"].astype(str)
        )
    return out.sort_values("sequence_id").reset_index(drop=True)


def select_entropy_adaptive_fusion(
    *,
    image_oof_probabilities: pd.DataFrame,
    nonimage_oof_probabilities: pd.DataFrame,
    train_labels: dict[str, str],
    image_predict_probabilities: pd.DataFrame,
    nonimage_predict_probabilities: pd.DataFrame,
    prior_image_weights: Iterable[float],
    adaptation_strengths: Iterable[float],
    entropy_powers: Iterable[float],
    selection_metric: str = "accuracy",
    probability_floor: float = 1e-9,
    output_dir: Path | None = None,
) -> EntropyAdaptiveFusionSelectionResult:
    """Select adaptive-fusion controls on OOF train predictions and freeze them."""

    metric = str(selection_metric)
    if metric not in SELECTION_METRICS:
        raise ValueError(f"unsupported selection metric: {metric!r}")
    labels = {str(key): str(value) for key, value in train_labels.items()}
    if not labels:
        raise ValueError("train_labels must not be empty")
    priors = _finite_grid(prior_image_weights, "prior_image_weights")
    strengths = _finite_grid(adaptation_strengths, "adaptation_strengths")
    powers = _finite_grid(entropy_powers, "entropy_powers")
    cv_rows: list[dict[str, Any]] = []
    for prior in priors:
        for strength in strengths:
            for power in powers:
                config = EntropyAdaptiveFusionConfig(
                    prior_image_weight=prior,
                    adaptation_strength=strength,
                    entropy_power=power,
                    probability_floor=probability_floor,
                )
                fused = fuse_entropy_adaptive_probabilities(
                    image_oof_probabilities,
                    nonimage_oof_probabilities,
                    config=config,
                    eval_labels=labels,
                )
                metrics = classification_metrics(fused)
                cv_rows.append({**asdict(config), **metrics})
    summary = pd.DataFrame.from_records(cv_rows)
    summary = _rank_cv_summary(summary, selection_metric=metric)
    selected_row = summary.iloc[0].to_dict()
    selected_config = EntropyAdaptiveFusionConfig(
        prior_image_weight=float(selected_row["prior_image_weight"]),
        adaptation_strength=float(selected_row["adaptation_strength"]),
        entropy_power=float(selected_row["entropy_power"]),
        probability_floor=float(selected_row["probability_floor"]),
        min_image_weight=float(selected_row["min_image_weight"]),
        max_image_weight=float(selected_row["max_image_weight"]),
    )
    selected_probabilities = fuse_entropy_adaptive_probabilities(
        image_predict_probabilities,
        nonimage_predict_probabilities,
        config=selected_config,
        class_source="train-oof-selected-entropy-adaptive-fusion",
    )
    manifest = {
        "schema": "raft-uav-mmuad-entropy-adaptive-fusion-v1",
        "selection_protocol": (
            "Out-of-fold train probabilities select all fusion controls; "
            "predict/evaluation labels are not used for selection."
        ),
        "selection_metric": metric,
        "train_sequence_count": int(summary.iloc[0]["sequence_count"]),
        "grid_size": int(len(summary)),
        "selected": _jsonable(selected_row),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / CV_SUMMARY_CSV, index=False)
        selected_probabilities.to_csv(
            output_dir / SELECTED_PROBABILITIES_CSV,
            index=False,
        )
        (output_dir / MANIFEST_JSON).write_text(
            json.dumps(_jsonable(manifest), indent=2),
            encoding="utf-8",
        )
    return EntropyAdaptiveFusionSelectionResult(
        cv_summary=summary,
        selected_probabilities=selected_probabilities,
        manifest=manifest,
    )


def classification_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Return accuracy, balanced accuracy, and multiclass log loss."""

    required = {"ground_truth_class", "predicted_class", *PROBABILITY_COLUMNS}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"probability rows missing evaluation columns: {missing}")
    scored = rows.loc[
        rows["ground_truth_class"].astype(str).isin(OFFICIAL_SEQUENCE_CLASS_LABELS)
    ].copy()
    if scored.empty:
        raise ValueError("no labeled probability rows are available for selection")
    truth = scored["ground_truth_class"].astype(str)
    predicted = scored["predicted_class"].astype(str)
    accuracy = float((truth == predicted).mean())
    recalls = []
    for label in sorted(truth.unique()):
        mask = truth == label
        recalls.append(float((predicted.loc[mask] == label).mean()))
    balanced_accuracy = float(np.mean(recalls))
    probabilities = scored.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    class_index = {label: index for index, label in enumerate(OFFICIAL_SEQUENCE_CLASS_LABELS)}
    indices = np.asarray([class_index[value] for value in truth], dtype=int)
    selected = probabilities[np.arange(len(scored)), indices]
    floor = np.finfo(float).tiny
    log_loss = float(-np.log(np.clip(selected, floor, 1.0)).mean())
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "log_loss": log_loss,
        "correct": int((truth == predicted).sum()),
        "sequence_count": int(len(scored)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "select sequence-specific entropy-adaptive MMUAD classifier fusion "
            "from out-of-fold train probabilities"
        )
    )
    parser.add_argument("--image-oof-probabilities", type=Path, required=True)
    parser.add_argument("--nonimage-oof-probabilities", type=Path, required=True)
    parser.add_argument("--image-predict-probabilities", type=Path, required=True)
    parser.add_argument("--nonimage-predict-probabilities", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-image-weight-grid", default="0:1:0.1")
    parser.add_argument("--adaptation-strength-grid", default="0:1:0.1")
    parser.add_argument("--entropy-power-grid", default="0.5,1,2")
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="accuracy")
    parser.add_argument("--probability-floor", type=float, default=1e-9)
    args = parser.parse_args(argv)

    result = select_entropy_adaptive_fusion(
        image_oof_probabilities=pd.read_csv(args.image_oof_probabilities),
        nonimage_oof_probabilities=pd.read_csv(args.nonimage_oof_probabilities),
        train_labels=_load_label_map(args.train_labels),
        image_predict_probabilities=pd.read_csv(args.image_predict_probabilities),
        nonimage_predict_probabilities=pd.read_csv(args.nonimage_predict_probabilities),
        prior_image_weights=_parse_grid(args.prior_image_weight_grid),
        adaptation_strengths=_parse_grid(args.adaptation_strength_grid),
        entropy_powers=_parse_grid(args.entropy_power_grid),
        selection_metric=args.selection_metric,
        probability_floor=float(args.probability_floor),
        output_dir=args.output_dir,
    )
    selected = result.manifest["selected"]
    print("mmuad_entropy_adaptive_fusion=ok")
    print(f"output_dir={Path(args.output_dir)}")
    print(f"selected_prior_image_weight={selected['prior_image_weight']}")
    print(f"selected_adaptation_strength={selected['adaptation_strength']}")
    print(f"selected_entropy_power={selected['entropy_power']}")
    print(f"selected_accuracy={selected['accuracy']}")
    print(f"selected_log_loss={selected['log_loss']}")
    return 0


def _fusion_diagnostics(
    image_vector: np.ndarray | None,
    nonimage_vector: np.ndarray | None,
    *,
    config: EntropyAdaptiveFusionConfig,
) -> dict[str, float]:
    if image_vector is None:
        return {
            "image_entropy": float("nan"),
            "nonimage_entropy": _normalized_entropy(nonimage_vector),
            "image_reliability": float("nan"),
            "nonimage_reliability": _reliability(nonimage_vector),
            "image_weight_prior": float(config.prior_image_weight),
            "image_weight_adaptive_target": 0.0,
            "image_weight_effective": 0.0,
        }
    if nonimage_vector is None:
        return {
            "image_entropy": _normalized_entropy(image_vector),
            "nonimage_entropy": float("nan"),
            "image_reliability": _reliability(image_vector),
            "nonimage_reliability": float("nan"),
            "image_weight_prior": float(config.prior_image_weight),
            "image_weight_adaptive_target": 1.0,
            "image_weight_effective": 1.0,
        }
    image_entropy = _normalized_entropy(image_vector)
    nonimage_entropy = _normalized_entropy(nonimage_vector)
    image_reliability = max(0.0, 1.0 - image_entropy)
    nonimage_reliability = max(0.0, 1.0 - nonimage_entropy)
    image_score = image_reliability ** float(config.entropy_power)
    nonimage_score = nonimage_reliability ** float(config.entropy_power)
    score_total = image_score + nonimage_score
    adaptive_target = 0.5 if score_total <= 0.0 else image_score / score_total
    effective = (
        (1.0 - float(config.adaptation_strength)) * float(config.prior_image_weight)
        + float(config.adaptation_strength) * adaptive_target
    )
    effective = float(
        np.clip(effective, config.min_image_weight, config.max_image_weight)
    )
    return {
        "image_entropy": image_entropy,
        "nonimage_entropy": nonimage_entropy,
        "image_reliability": image_reliability,
        "nonimage_reliability": nonimage_reliability,
        "image_weight_prior": float(config.prior_image_weight),
        "image_weight_adaptive_target": float(adaptive_target),
        "image_weight_effective": effective,
    }


def _normalized_probability_index(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", *PROBABILITY_COLUMNS]).set_index(
            "sequence_id",
            drop=False,
        )
    if "sequence_id" not in rows.columns:
        raise ValueError(f"{name} must contain sequence_id")
    missing = sorted(set(PROBABILITY_COLUMNS).difference(rows.columns))
    if missing:
        raise ValueError(f"{name} missing probability columns: {missing}")
    out = rows.loc[:, ["sequence_id", *PROBABILITY_COLUMNS]].copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    if out["sequence_id"].duplicated().any():
        duplicates = sorted(out.loc[out["sequence_id"].duplicated(), "sequence_id"].unique())
        raise ValueError(f"{name} has duplicate sequence rows: {duplicates[:10]}")
    values = out.loc[:, PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    numeric = values.to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0.0).any():
        raise ValueError(f"{name} probabilities must be finite and non-negative")
    totals = numeric.sum(axis=1)
    if (totals <= 0.0).any():
        bad = np.flatnonzero(totals <= 0.0).tolist()
        raise ValueError(f"{name} probability rows must have positive mass: {bad[:10]}")
    out.loc[:, PROBABILITY_COLUMNS] = numeric / totals[:, None]
    return out.set_index("sequence_id", drop=False)


def _probability_vector(rows: pd.DataFrame, sequence_id: str) -> np.ndarray:
    values = rows.loc[sequence_id, list(PROBABILITY_COLUMNS)]
    return np.asarray(values, dtype=float)


def _normalize_vector(values: np.ndarray, *, probability_floor: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    vector = np.maximum(vector, float(probability_floor))
    return vector / float(vector.sum())


def _normalized_entropy(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    vector = np.asarray(values, dtype=float)
    positive = vector > 0.0
    entropy = -float(np.sum(vector[positive] * np.log(vector[positive])))
    return entropy / float(np.log(len(OFFICIAL_SEQUENCE_CLASS_LABELS)))


def _reliability(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    return max(0.0, 1.0 - _normalized_entropy(values))


def _rank_cv_summary(rows: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    if selection_metric == "log_loss":
        columns = [
            "log_loss",
            "accuracy",
            "balanced_accuracy",
            "adaptation_strength",
            "entropy_power",
            "prior_image_weight",
        ]
        ascending = [True, False, False, True, True, True]
    elif selection_metric == "balanced_accuracy":
        columns = [
            "balanced_accuracy",
            "accuracy",
            "log_loss",
            "adaptation_strength",
            "entropy_power",
            "prior_image_weight",
        ]
        ascending = [False, False, True, True, True, True]
    else:
        columns = [
            "accuracy",
            "balanced_accuracy",
            "log_loss",
            "adaptation_strength",
            "entropy_power",
            "prior_image_weight",
        ]
        ascending = [False, False, True, True, True, True]
    return rows.sort_values(columns, ascending=ascending).reset_index(drop=True)


def _validate_config(config: EntropyAdaptiveFusionConfig) -> None:
    bounded = {
        "prior_image_weight": config.prior_image_weight,
        "adaptation_strength": config.adaptation_strength,
        "min_image_weight": config.min_image_weight,
        "max_image_weight": config.max_image_weight,
    }
    for name, value in bounded.items():
        numeric = _finite_float(value, name=name)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    power = _finite_float(config.entropy_power, name="entropy_power")
    if power <= 0.0:
        raise ValueError("entropy_power must be positive")
    floor = _finite_float(config.probability_floor, name="probability_floor")
    if floor <= 0.0:
        raise ValueError("probability_floor must be positive")
    if float(config.min_image_weight) > float(config.max_image_weight):
        raise ValueError("min_image_weight must not exceed max_image_weight")


def _finite_grid(values: Iterable[float], name: str) -> list[float]:
    result = [_finite_float(value, name=name) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _finite_float(value: float, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _parse_grid(text: str) -> list[float]:
    value = str(text).strip()
    if ":" not in value:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    start_text, stop_text, step_text = value.split(":", 2)
    start = float(start_text)
    stop = float(stop_text)
    step = float(step_text)
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    values = []
    current = start
    while current <= stop + step / 10.0:
        values.append(round(float(current), 10))
        current += step
    return values


def _load_label_map(path: Path) -> dict[str, str]:
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    sequence_column = next(
        (name for name in ("sequence_id", "Sequence", "sequence") if name in rows.columns),
        None,
    )
    label_column = next(
        (
            name
            for name in ("uav_type", "Classification", "class", "label")
            if name in rows.columns
        ),
        None,
    )
    if sequence_column is None or label_column is None:
        raise ValueError("train label CSV must contain sequence and class columns")
    return dict(
        zip(
            rows[sequence_column].astype(str),
            rows[label_column].astype(str),
            strict=True,
        )
    )


def _empty_output() -> pd.DataFrame:
    columns = [
        "sequence_id",
        "image_available",
        "nonimage_available",
        "image_entropy",
        "nonimage_entropy",
        "image_reliability",
        "nonimage_reliability",
        "image_weight_prior",
        "image_weight_adaptive_target",
        "image_weight_effective",
        *PROBABILITY_COLUMNS,
        "predicted_class",
        "class_source",
    ]
    return pd.DataFrame(columns=columns)


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
