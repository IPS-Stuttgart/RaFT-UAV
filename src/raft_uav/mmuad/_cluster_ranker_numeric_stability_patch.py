"""Keep cluster-ranker distances and normalization stable for finite extremes."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import math
from typing import Any

import numpy as np
import pandas as pd

_LABEL_PATCH_MARKER = "_raft_uav_cluster_ranker_numeric_stability"
_STANDARDIZE_PATCH_MARKER = "_raft_uav_cluster_ranker_standardize_numeric_stability"
_TRAIN_PATCH_MARKER = "_raft_uav_cluster_ranker_train_numeric_stability"
_PREDICT_PATCH_MARKER = "_raft_uav_cluster_ranker_predict_numeric_stability"


def _scaled_mean(values: np.ndarray) -> float:
    """Return a finite mean without requiring the unscaled sum to be finite."""

    if values.size == 0:
        return 0.0
    scale = max(abs(float(value)) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = math.fsum(float(value) / scale for value in values) / len(values)
    normalized = min(1.0, max(-1.0, normalized))
    return scale * normalized


def _scaled_population_std(values: np.ndarray, mean: float) -> float:
    """Return population standard deviation without overflowing deviations."""

    if values.size == 0:
        return 0.0
    scale = max(
        abs(float(mean)),
        max(abs(float(value)) for value in values),
    )
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return float("inf")
    normalized_mean = float(mean) / scale
    normalized_variance = math.fsum(
        (float(value) / scale - normalized_mean) ** 2
        for value in values
    ) / len(values)
    normalized_variance = min(1.0, max(0.0, normalized_variance))
    return scale * math.sqrt(normalized_variance)


def _stable_standardize_training_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the legacy transform while repairing overflowed column statistics."""

    values = np.asarray(matrix, dtype=float)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=0)
    with np.errstate(all="ignore"):
        means = np.divide(
            np.where(finite_mask, values, 0.0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1], dtype=float),
            where=counts > 0,
        )
        filled = np.where(finite_mask, values, means)
        raw_scales = np.nanstd(filled, axis=0)
    bad_columns = (counts > 0) & (
        ~np.isfinite(means) | ~np.isfinite(raw_scales)
    )
    if bool(bad_columns.any()):
        means = means.copy()
        filled = filled.copy()
        raw_scales = raw_scales.copy()
        for column in np.flatnonzero(bad_columns):
            finite_values = values[finite_mask[:, column], column]
            mean = _scaled_mean(finite_values)
            means[column] = mean
            filled[:, column] = np.where(
                finite_mask[:, column],
                values[:, column],
                mean,
            )
            raw_scales[column] = _scaled_population_std(
                filled[:, column],
                mean,
            )
    scales = np.where(
        np.isfinite(raw_scales) & (raw_scales > 1.0e-9),
        raw_scales,
        1.0,
    )
    return filled, means, scales


def _stable_center_scale(
    matrix: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Repair centering overflow while preserving ordinary standardized values."""

    values = np.asarray(matrix, dtype=float)
    centers = np.asarray(means, dtype=float)
    denominators = np.asarray(scales, dtype=float)
    with np.errstate(all="ignore"):
        standardized = (values - centers) / denominators
    if bool(np.isfinite(standardized).all()):
        return standardized
    with np.errstate(all="ignore"):
        scaled_first = values / denominators - centers / denominators
    repair = ~np.isfinite(standardized) & np.isfinite(scaled_first)
    if bool(repair.any()):
        standardized = standardized.copy()
        standardized[repair] = scaled_first[repair]
    return standardized


def _train_cluster_ranker_stably(
    cluster_ranker: Any,
    features: pd.DataFrame,
    *,
    model_type: str = "logistic",
    target_column: str = "good_cluster",
    learning_rate: float = 0.05,
    iterations: int = 600,
    l2: float = 1.0e-3,
    random_state: int = 13,
    n_estimators: int = 200,
    score_distance_scale_m: float = 10.0,
) -> Any:
    """Run the legacy trainer with overflow-safe centering and scaling."""

    model_type = str(model_type)
    actual_target = cluster_ranker._actual_target_column(
        features,
        model_type=model_type,
        target_column=target_column,
    )
    rows = features.loc[features[actual_target].notna()].copy()
    if rows.empty:
        raise ValueError(f"no rows with target column {actual_target!r}")
    source_values = sorted(rows["source"].fillna("").astype(str).unique())
    feature_columns = cluster_ranker._ranker_feature_columns(rows, source_values)
    matrix = cluster_ranker._feature_matrix(
        rows,
        feature_columns,
        source_values=source_values,
    )
    matrix, means, scales = cluster_ranker._standardize_training_matrix(matrix)
    x = _stable_center_scale(matrix, means, scales)
    if model_type != "logistic":
        return cluster_ranker._train_sklearn_cluster_ranker(
            x,
            rows,
            model_type=model_type,
            target_column=actual_target,
            feature_columns=feature_columns,
            feature_means=means,
            feature_scales=scales,
            source_values=source_values,
            random_state=random_state,
            n_estimators=n_estimators,
            score_distance_scale_m=score_distance_scale_m,
        )
    y = rows[actual_target].astype(bool).astype(float).to_numpy()
    positive_rate = float(np.mean(y))
    if positive_rate <= 0.0 or positive_rate >= 1.0:
        return cluster_ranker.ClusterRankerModel(
            model_type="constant-logistic",
            feature_columns=feature_columns,
            feature_means=means.tolist(),
            feature_scales=scales.tolist(),
            weights=[0.0] * len(feature_columns),
            bias=cluster_ranker._logit(
                np.clip(positive_rate, 1.0e-6, 1.0 - 1.0e-6)
            ),
            source_values=source_values,
            constant_score=positive_rate,
            target_column=actual_target,
            score_distance_scale_m=float(score_distance_scale_m),
        )
    weights = np.zeros(x.shape[1], dtype=float)
    bias = cluster_ranker._logit(positive_rate)
    for _ in range(max(int(iterations), 1)):
        logits = x @ weights + bias
        pred = cluster_ranker._sigmoid(logits)
        error = pred - y
        weights -= float(learning_rate) * (
            (x.T @ error) / len(y) + float(l2) * weights
        )
        bias -= float(learning_rate) * float(np.mean(error))
    return cluster_ranker.ClusterRankerModel(
        model_type="logistic",
        feature_columns=feature_columns,
        feature_means=means.tolist(),
        feature_scales=scales.tolist(),
        weights=weights.tolist(),
        bias=float(bias),
        source_values=source_values,
        constant_score=None,
        target_column=actual_target,
        score_distance_scale_m=float(score_distance_scale_m),
    )


def _predict_cluster_scores_stably(
    cluster_ranker: Any,
    features: pd.DataFrame,
    model: Any,
) -> np.ndarray:
    """Predict with overflow-safe centering of finite feature extremes."""

    if features.empty:
        return np.asarray([], dtype=float)
    matrix = cluster_ranker._feature_matrix(
        features,
        model.feature_columns,
        source_values=model.source_values,
    )
    means = np.asarray(model.feature_means, dtype=float)
    scales = np.asarray(model.feature_scales, dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, means)
    x = _stable_center_scale(matrix, means, scales)
    if model.sklearn_estimator_base64:
        estimator = cluster_ranker._decode_sklearn_estimator(
            model.sklearn_estimator_base64
        )
        if model.score_transform == "inverse-distance":
            distances = np.asarray(estimator.predict(x), dtype=float)
            distances = np.maximum(
                np.nan_to_num(distances, nan=1.0e6),
                0.0,
            )
            scale = max(float(model.score_distance_scale_m), 1.0e-6)
            return 1.0 / (1.0 + distances / scale)
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(x)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return np.asarray(probabilities[:, 1], dtype=float)
            return np.asarray(probabilities).reshape(-1).astype(float)
        if hasattr(estimator, "decision_function"):
            return cluster_ranker._sigmoid(
                np.asarray(estimator.decision_function(x), dtype=float)
            )
        return np.asarray(estimator.predict(x), dtype=float)
    if model.constant_score is not None:
        return np.full(len(features), float(model.constant_score), dtype=float)
    logits = x @ np.asarray(model.weights, dtype=float) + float(model.bias)
    return cluster_ranker._sigmoid(logits)


def _needs_distance_repair(legacy: float, stable: float) -> bool:
    if math.isfinite(stable) and not math.isfinite(legacy):
        return True
    return legacy == 0.0 and stable > 0.0


def _repair_truth_distances(
    labeled: pd.DataFrame,
    features: pd.DataFrame,
    truth: pd.DataFrame,
    cluster_ranker: Any,
    *,
    good_threshold_m: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Repair only norm overflow/underflow while preserving ordinary outputs."""

    if labeled.empty or "truth_matched" not in labeled.columns:
        return labeled
    authoritative = cluster_ranker._authoritative_truth_rows(truth)
    if authoritative.empty:
        return labeled
    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in authoritative.groupby(
            "sequence_id",
            sort=True,
        )
    }
    matched = (
        labeled["truth_matched"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    legacy_3d = pd.to_numeric(
        labeled["truth_distance_3d_m"],
        errors="coerce",
    ).to_numpy(float)
    legacy_2d = pd.to_numeric(
        labeled["truth_distance_2d_m"],
        errors="coerce",
    ).to_numpy(float)
    feature_rows = pd.DataFrame(features)
    time_gate = float(max_truth_time_delta_s)
    distance_gate = float(good_threshold_m)
    thresholds = {
        "good_cluster_2m": 2.0,
        "good_cluster_5m": 5.0,
        "good_cluster_10m": 10.0,
        "good_cluster_20m": 20.0,
        "good_cluster": distance_gate,
    }

    for position in np.flatnonzero(matched):
        row = feature_rows.iloc[int(position)]
        seq_truth = truth_by_sequence.get(str(row["sequence_id"]))
        if seq_truth is None or seq_truth.empty:
            continue
        try:
            row_time = float(row["time_s"])
            truth_times = [
                float(value) for value in seq_truth["time_s"].tolist()
            ]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(row_time):
            continue
        if not all(math.isfinite(value) for value in truth_times):
            continue
        truth_position = min(
            range(len(truth_times)),
            key=lambda index: abs(truth_times[index] - row_time),
        )
        time_delta = row_time - truth_times[truth_position]
        if abs(time_delta) > time_gate:
            continue
        truth_row = seq_truth.iloc[truth_position]
        try:
            residual = tuple(
                float(row[column]) - float(truth_row[column])
                for column in ("x_m", "y_m", "z_m")
            )
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in residual):
            continue
        stable_2d = math.hypot(residual[0], residual[1])
        stable_3d = math.hypot(*residual)
        repair_2d = _needs_distance_repair(
            float(legacy_2d[position]),
            stable_2d,
        )
        repair_3d = _needs_distance_repair(
            float(legacy_3d[position]),
            stable_3d,
        )
        if not (repair_2d or repair_3d):
            continue
        index = int(position)
        if repair_2d:
            column = labeled.columns.get_loc("truth_distance_2d_m")
            labeled.iloc[index, column] = stable_2d
        if repair_3d:
            column = labeled.columns.get_loc("truth_distance_3d_m")
            labeled.iloc[index, column] = stable_3d
            if "truth_vertical_error_m" in labeled.columns:
                column = labeled.columns.get_loc("truth_vertical_error_m")
                labeled.iloc[index, column] = abs(residual[2])
            for column_name, threshold in thresholds.items():
                if column_name not in labeled.columns:
                    continue
                column = labeled.columns.get_loc(column_name)
                labeled.iloc[index, column] = stable_3d <= threshold
    return labeled


def install() -> None:
    """Install finite-extreme stability repairs for cluster-ranker helpers."""

    from raft_uav.mmuad import cluster_ranker

    previous_label: Callable[..., pd.DataFrame]
    previous_label = cluster_ranker.label_cluster_features_against_truth
    if not getattr(previous_label, _LABEL_PATCH_MARKER, False):

        @wraps(previous_label)
        def stable_label_cluster_features_against_truth(
            features: pd.DataFrame,
            truth: pd.DataFrame,
            *,
            good_threshold_m: float = 5.0,
            max_truth_time_delta_s: float = 0.5,
        ) -> pd.DataFrame:
            with np.errstate(all="ignore"):
                labeled = previous_label(
                    features,
                    truth,
                    good_threshold_m=good_threshold_m,
                    max_truth_time_delta_s=max_truth_time_delta_s,
                )
            return _repair_truth_distances(
                labeled,
                features,
                truth,
                cluster_ranker,
                good_threshold_m=good_threshold_m,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )

        setattr(
            stable_label_cluster_features_against_truth,
            _LABEL_PATCH_MARKER,
            True,
        )
        setattr(
            stable_label_cluster_features_against_truth,
            "_raft_uav_previous",
            previous_label,
        )
        cluster_ranker.label_cluster_features_against_truth = (
            stable_label_cluster_features_against_truth
        )
        cluster_ranker._IMPL.label_cluster_features_against_truth = (
            stable_label_cluster_features_against_truth
        )

    previous_standardize = cluster_ranker._standardize_training_matrix
    if not getattr(previous_standardize, _STANDARDIZE_PATCH_MARKER, False):

        @wraps(previous_standardize)
        def stable_standardize_training_matrix(
            matrix: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _stable_standardize_training_matrix(matrix)

        setattr(
            stable_standardize_training_matrix,
            _STANDARDIZE_PATCH_MARKER,
            True,
        )
        setattr(
            stable_standardize_training_matrix,
            "_raft_uav_previous",
            previous_standardize,
        )
        cluster_ranker._standardize_training_matrix = (
            stable_standardize_training_matrix
        )
        cluster_ranker._IMPL._standardize_training_matrix = (
            stable_standardize_training_matrix
        )

    previous_train = cluster_ranker._LEGACY_TRAIN_CLUSTER_RANKER
    if not getattr(previous_train, _TRAIN_PATCH_MARKER, False):

        @wraps(previous_train)
        def stable_train_cluster_ranker(
            features: pd.DataFrame,
            *,
            model_type: str = "logistic",
            target_column: str = "good_cluster",
            learning_rate: float = 0.05,
            iterations: int = 600,
            l2: float = 1.0e-3,
            random_state: int = 13,
            n_estimators: int = 200,
            score_distance_scale_m: float = 10.0,
        ) -> Any:
            return _train_cluster_ranker_stably(
                cluster_ranker,
                features,
                model_type=model_type,
                target_column=target_column,
                learning_rate=learning_rate,
                iterations=iterations,
                l2=l2,
                random_state=random_state,
                n_estimators=n_estimators,
                score_distance_scale_m=score_distance_scale_m,
            )

        setattr(stable_train_cluster_ranker, _TRAIN_PATCH_MARKER, True)
        setattr(
            stable_train_cluster_ranker,
            "_raft_uav_previous",
            previous_train,
        )
        cluster_ranker._LEGACY_TRAIN_CLUSTER_RANKER = (
            stable_train_cluster_ranker
        )

    previous_predict = cluster_ranker.predict_cluster_scores
    if not getattr(previous_predict, _PREDICT_PATCH_MARKER, False):

        @wraps(previous_predict)
        def stable_predict_cluster_scores(
            features: pd.DataFrame,
            model: Any,
        ) -> np.ndarray:
            return _predict_cluster_scores_stably(
                cluster_ranker,
                features,
                model,
            )

        setattr(stable_predict_cluster_scores, _PREDICT_PATCH_MARKER, True)
        setattr(
            stable_predict_cluster_scores,
            "_raft_uav_previous",
            previous_predict,
        )
        cluster_ranker.predict_cluster_scores = stable_predict_cluster_scores
        cluster_ranker._IMPL.predict_cluster_scores = (
            stable_predict_cluster_scores
        )
