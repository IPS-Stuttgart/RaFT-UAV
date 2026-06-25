"""Train and apply learned per-candidate uncertainty for MMUAD pose inference.

Candidate-mixture tracking benefits from a candidate-specific measurement scale
rather than one fixed covariance for every cluster. This module learns expected
candidate position error from training truth and existing cluster-ranker
features, then writes a clipped ``predicted_sigma_m`` column for robust
mixture-MAP or tracker experiments.

Class-probability, source, and candidate-branch interactions are consumed when
they are already present as numeric ``image_*`` feature columns. Applying a
saved model does not require validation or test truth.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.cluster_ranker import (
    BASE_CLUSTER_FEATURE_COLUMNS,
    build_cluster_feature_table,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

MODEL_TYPES = ("ridge", "random-forest", "hist-gradient-boosting")
TARGET_TRANSFORMS = ("identity", "log1p")
_SOURCE_PREFIX = "source=="


@dataclass(frozen=True)
class CandidateUncertaintyModel:
    """Portable model for candidate error-scale prediction."""

    model_type: str
    feature_columns: list[str]
    feature_means: list[float]
    feature_scales: list[float]
    source_values: list[str]
    target_transform: str
    sigma_min_m: float
    sigma_max_m: float
    fallback_sigma_m: float
    weights: list[float]
    bias: float
    sklearn_estimator_base64: str | None = None


def train_candidate_uncertainty(
    features: pd.DataFrame,
    *,
    model_type: str = "hist-gradient-boosting",
    target_transform: str = "log1p",
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 30.0,
    ridge_alpha: float = 1.0,
    random_state: int = 13,
    n_estimators: int = 300,
) -> CandidateUncertaintyModel:
    """Fit expected 3D candidate error from train-labeled feature rows."""

    model_type = str(model_type)
    target_transform = str(target_transform)
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unsupported uncertainty model_type={model_type!r}")
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(f"unsupported target_transform={target_transform!r}")
    if float(sigma_min_m) <= 0.0 or float(sigma_max_m) < float(sigma_min_m):
        raise ValueError("sigma bounds must satisfy 0 < sigma_min_m <= sigma_max_m")

    rows = pd.DataFrame(features).copy()
    if "truth_distance_3d_m" not in rows.columns:
        raise ValueError("uncertainty training requires truth_distance_3d_m labels")
    target_distance = pd.to_numeric(rows["truth_distance_3d_m"], errors="coerce")
    finite = np.isfinite(target_distance.to_numpy(float))
    rows = rows.loc[finite].reset_index(drop=True)
    target_distance = target_distance.loc[finite].reset_index(drop=True)
    if rows.empty:
        raise ValueError("no finite truth-distance rows for uncertainty training")

    source_series = rows.get("source", pd.Series("", index=rows.index))
    source_values = sorted(source_series.fillna("").astype(str).unique())
    feature_columns = _uncertainty_feature_columns(rows, source_values=source_values)
    matrix = _feature_matrix(rows, feature_columns)
    matrix, means, scales = _standardize_training_matrix(matrix)
    x = (matrix - means) / scales
    y_distance = np.maximum(target_distance.to_numpy(float), 0.0)
    y = _forward_target(y_distance, transform=target_transform)
    fallback_sigma = float(np.clip(np.median(y_distance), sigma_min_m, sigma_max_m))

    if model_type == "ridge":
        y_mean = float(np.mean(y))
        centered = y - y_mean
        gram = x.T @ x
        regularized = gram + float(max(ridge_alpha, 0.0)) * np.eye(x.shape[1])
        weights = np.linalg.pinv(regularized) @ x.T @ centered
        return CandidateUncertaintyModel(
            model_type=model_type,
            feature_columns=feature_columns,
            feature_means=means.tolist(),
            feature_scales=scales.tolist(),
            source_values=source_values,
            target_transform=target_transform,
            sigma_min_m=float(sigma_min_m),
            sigma_max_m=float(sigma_max_m),
            fallback_sigma_m=fallback_sigma,
            weights=weights.astype(float).tolist(),
            bias=y_mean,
        )

    estimator = _make_sklearn_regressor(
        model_type=model_type,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    estimator.fit(x, y)
    return CandidateUncertaintyModel(
        model_type=model_type,
        feature_columns=feature_columns,
        feature_means=means.tolist(),
        feature_scales=scales.tolist(),
        source_values=source_values,
        target_transform=target_transform,
        sigma_min_m=float(sigma_min_m),
        sigma_max_m=float(sigma_max_m),
        fallback_sigma_m=fallback_sigma,
        weights=[0.0] * len(feature_columns),
        bias=0.0,
        sklearn_estimator_base64=_encode_estimator(estimator),
    )


def predict_candidate_sigma(
    features: pd.DataFrame,
    model: CandidateUncertaintyModel,
) -> np.ndarray:
    """Predict clipped 3D error scales for candidate feature rows."""

    rows = pd.DataFrame(features)
    if rows.empty:
        return np.asarray([], dtype=float)
    matrix = _feature_matrix(rows, model.feature_columns)
    means = np.asarray(model.feature_means, dtype=float)
    scales = np.asarray(model.feature_scales, dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, means)
    x = (matrix - means) / scales
    if model.sklearn_estimator_base64:
        estimator = _decode_estimator(model.sklearn_estimator_base64)
        transformed = np.asarray(estimator.predict(x), dtype=float)
    else:
        transformed = x @ np.asarray(model.weights, dtype=float) + float(model.bias)
    distance = _inverse_target(transformed, transform=model.target_transform)
    distance = np.nan_to_num(
        distance,
        nan=float(model.fallback_sigma_m),
        posinf=float(model.sigma_max_m),
        neginf=float(model.sigma_min_m),
    )
    return np.clip(distance, float(model.sigma_min_m), float(model.sigma_max_m))


def apply_candidate_uncertainty(
    candidates: CandidateFrame | pd.DataFrame,
    model: CandidateUncertaintyModel,
    *,
    output_column: str = "predicted_sigma_m",
    replace_covariance: bool = False,
    z_scale: float = 1.0,
) -> CandidateFrame:
    """Attach predicted uncertainty to inference candidates without truth."""

    features = build_cluster_feature_table(candidates)
    if features.empty:
        return CandidateFrame(normalize_candidate_columns(features))
    rows = features.copy()
    rows[output_column] = predict_candidate_sigma(rows, model)
    if replace_covariance:
        rows["raw_std_xy_m"] = pd.to_numeric(rows.get("std_xy_m"), errors="coerce")
        rows["raw_std_z_m"] = pd.to_numeric(rows.get("std_z_m"), errors="coerce")
        rows["std_xy_m"] = rows[output_column]
        rows["std_z_m"] = rows[output_column] * float(z_scale)
    return CandidateFrame(normalize_candidate_columns(rows))


def candidate_uncertainty_training_summary(
    features: pd.DataFrame,
    model: CandidateUncertaintyModel,
) -> dict[str, Any]:
    """Return compact in-sample diagnostics for model provenance."""

    if "truth_distance_3d_m" not in features.columns:
        return {"row_count": 0}
    truth = pd.to_numeric(features["truth_distance_3d_m"], errors="coerce")
    predicted = pd.Series(predict_candidate_sigma(features, model), index=features.index)
    finite = truth.notna() & predicted.notna()
    if not finite.any():
        return {"row_count": 0}
    residual = predicted.loc[finite].to_numpy(float) - truth.loc[finite].to_numpy(float)
    truth_values = truth.loc[finite].to_numpy(float)
    predicted_values = predicted.loc[finite].to_numpy(float)
    correlation = float("nan")
    if (
        len(truth_values) >= 2
        and np.std(truth_values) > 0.0
        and np.std(predicted_values) > 0.0
    ):
        correlation = float(np.corrcoef(truth_values, predicted_values)[0, 1])
    return {
        "row_count": int(finite.sum()),
        "mae_m": float(np.mean(np.abs(residual))),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "p95_abs_error_m": float(np.quantile(np.abs(residual), 0.95)),
        "truth_mean_m": float(np.mean(truth_values)),
        "predicted_mean_m": float(np.mean(predicted_values)),
        "correlation": correlation,
        "model_type": model.model_type,
        "target_transform": model.target_transform,
        "feature_count": len(model.feature_columns),
        "sigma_min_m": model.sigma_min_m,
        "sigma_max_m": model.sigma_max_m,
    }


def save_candidate_uncertainty_model(
    model: CandidateUncertaintyModel,
    path: Path,
) -> Path:
    """Write an uncertainty model JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")
    return path


def load_candidate_uncertainty_model(path: Path) -> CandidateUncertaintyModel:
    """Read an uncertainty model JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return CandidateUncertaintyModel(
        model_type=str(payload["model_type"]),
        feature_columns=[str(value) for value in payload["feature_columns"]],
        feature_means=[float(value) for value in payload["feature_means"]],
        feature_scales=[float(value) for value in payload["feature_scales"]],
        source_values=[str(value) for value in payload.get("source_values", [])],
        target_transform=str(payload.get("target_transform", "log1p")),
        sigma_min_m=float(payload.get("sigma_min_m", 1.0)),
        sigma_max_m=float(payload.get("sigma_max_m", 30.0)),
        fallback_sigma_m=float(payload.get("fallback_sigma_m", 10.0)),
        weights=[float(value) for value in payload.get("weights", [])],
        bias=float(payload.get("bias", 0.0)),
        sklearn_estimator_base64=payload.get("sklearn_estimator_base64"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_uncertainty",
        description="train or apply an MMUAD per-candidate uncertainty model",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--candidates-csv", type=Path, required=True)
    train.add_argument("--truth-csv", type=Path, required=True)
    train.add_argument("--model-json", type=Path, required=True)
    train.add_argument("--features-csv", type=Path)
    train.add_argument("--summary-json", type=Path)
    train.add_argument("--model-type", choices=MODEL_TYPES, default="hist-gradient-boosting")
    train.add_argument("--target-transform", choices=TARGET_TRANSFORMS, default="log1p")
    train.add_argument("--sigma-min-m", type=float, default=1.0)
    train.add_argument("--sigma-max-m", type=float, default=30.0)
    train.add_argument("--ridge-alpha", type=float, default=1.0)
    train.add_argument("--random-state", type=int, default=13)
    train.add_argument("--n-estimators", type=int, default=300)
    train.add_argument("--max-truth-time-delta-s", type=float, default=0.5)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--candidates-csv", type=Path, required=True)
    apply_parser.add_argument("--model-json", type=Path, required=True)
    apply_parser.add_argument("--output-csv", type=Path, required=True)
    apply_parser.add_argument("--output-column", default="predicted_sigma_m")
    apply_parser.add_argument("--replace-covariance", action="store_true")
    apply_parser.add_argument("--z-scale", type=float, default=1.0)

    args = parser.parse_args(argv)
    if args.command == "train":
        candidates = load_candidate_file(args.candidates_csv)
        truth = load_evaluation_truth_file(args.truth_csv).rows
        features = build_cluster_feature_table(
            candidates,
            truth=truth,
            max_truth_time_delta_s=float(args.max_truth_time_delta_s),
        )
        model = train_candidate_uncertainty(
            features,
            model_type=args.model_type,
            target_transform=args.target_transform,
            sigma_min_m=args.sigma_min_m,
            sigma_max_m=args.sigma_max_m,
            ridge_alpha=args.ridge_alpha,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        save_candidate_uncertainty_model(model, args.model_json)
        if args.features_csv is not None:
            args.features_csv.parent.mkdir(parents=True, exist_ok=True)
            features.to_csv(args.features_csv, index=False)
        summary = candidate_uncertainty_training_summary(features, model)
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("mmuad_candidate_uncertainty_train=ok")
        print(f"model_json={args.model_json}")
        print(f"training_rows={summary.get('row_count', 0)}")
        return 0

    model = load_candidate_uncertainty_model(args.model_json)
    scored = apply_candidate_uncertainty(
        load_candidate_file(args.candidates_csv),
        model,
        output_column=args.output_column,
        replace_covariance=args.replace_covariance,
        z_scale=args.z_scale,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.rows.to_csv(args.output_csv, index=False)
    print("mmuad_candidate_uncertainty_apply=ok")
    print(f"output_csv={args.output_csv}")
    print(f"output_rows={len(scored.rows)}")
    return 0


def _uncertainty_feature_columns(
    rows: pd.DataFrame,
    *,
    source_values: list[str],
) -> list[str]:
    columns: list[str] = []
    for column in BASE_CLUSTER_FEATURE_COLUMNS:
        if column in rows.columns and _has_numeric_value(rows[column]):
            columns.append(column)
    extra_prefixes = (
        "image_",
        "candidate_reservoir_",
        "candidate_diversity_",
        "dynamic_",
    )
    for column in sorted(str(value) for value in rows.columns):
        if column in columns or not column.startswith(extra_prefixes):
            continue
        if _has_numeric_value(rows[column]):
            columns.append(column)
    columns.extend(f"{_SOURCE_PREFIX}{source}" for source in source_values)
    return columns


def _feature_matrix(rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    source = rows.get("source", pd.Series("", index=rows.index)).fillna("").astype(str)
    values: list[np.ndarray] = []
    for column in feature_columns:
        if column.startswith(_SOURCE_PREFIX):
            source_value = column[len(_SOURCE_PREFIX) :]
            values.append((source == source_value).astype(float).to_numpy())
        else:
            default = pd.Series(np.nan, index=rows.index)
            numeric = pd.to_numeric(rows.get(column, default), errors="coerce")
            values.append(numeric.to_numpy(float))
    if not values:
        return np.empty((len(rows), 0), dtype=float)
    return np.column_stack(values)


def _standardize_training_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.shape[1] == 0:
        raise ValueError("no numeric features available for uncertainty training")
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    means = np.divide(
        np.where(finite, matrix, 0.0).sum(axis=0),
        counts,
        out=np.zeros(matrix.shape[1], dtype=float),
        where=counts > 0,
    )
    filled = np.where(finite, matrix, means)
    scales = np.std(filled, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-9), scales, 1.0)
    return filled, means, scales


def _has_numeric_value(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(float)).any())


def _forward_target(values: np.ndarray, *, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.log1p(np.maximum(np.asarray(values, dtype=float), 0.0))
    raise ValueError(f"unsupported target transform: {transform}")


def _inverse_target(values: np.ndarray, *, transform: str) -> np.ndarray:
    if transform == "identity":
        return np.maximum(np.asarray(values, dtype=float), 0.0)
    if transform == "log1p":
        return np.maximum(np.expm1(np.asarray(values, dtype=float)), 0.0)
    raise ValueError(f"unsupported target transform: {transform}")


def _make_sklearn_regressor(
    *,
    model_type: str,
    random_state: int,
    n_estimators: int,
) -> Any:
    try:
        if model_type == "random-forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=max(int(n_estimators), 1),
                random_state=int(random_state),
                min_samples_leaf=2,
                n_jobs=-1,
            )
        if model_type == "hist-gradient-boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                random_state=int(random_state),
                max_iter=max(int(n_estimators), 1),
                learning_rate=0.05,
                l2_regularization=0.01,
                loss="squared_error",
            )
    except ImportError as exc:
        raise ValueError(
            f"{model_type} requires scikit-learn; install sklearn or use --model-type ridge"
        ) from exc
    raise ValueError(f"unsupported sklearn uncertainty model_type={model_type!r}")


def _encode_estimator(estimator: Any) -> str:
    return base64.b64encode(pickle.dumps(estimator)).decode("ascii")


def _decode_estimator(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
