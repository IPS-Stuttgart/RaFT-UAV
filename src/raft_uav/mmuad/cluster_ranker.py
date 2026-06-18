"""Supervised point-cloud cluster ranking for MMUAD candidates."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import merge_candidate_frames
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export


BASE_CLUSTER_FEATURE_COLUMNS = (
    "x_m",
    "y_m",
    "z_m",
    "confidence",
    "cluster_point_count",
    "cluster_extent_x_m",
    "cluster_extent_y_m",
    "cluster_extent_z_m",
    "cluster_extent_xy_m",
    "cluster_extent_3d_m",
    "cluster_bbox_volume_m3",
    "cluster_density_points_per_m3",
    "cluster_range_xy_m",
    "cluster_range_3d_m",
    "cluster_height_m",
    "frame_candidate_count",
    "frame_source_candidate_count",
    "frame_source_fraction",
    "frame_rank_confidence_desc",
    "frame_rank_point_count_desc",
    "frame_rank_density_desc",
    "frame_rank_range_3d_asc",
    "frame_rank_height_abs_asc",
    "source_frame_rank_confidence_desc",
    "source_frame_rank_point_count_desc",
    "source_frame_rank_density_desc",
    "source_frame_rank_range_3d_asc",
    "nearest_cross_sensor_distance_m",
    "nearest_cross_sensor_score",
    "cross_sensor_neighbor_count",
    "prev_same_source_distance_m",
    "prev_same_source_dt_s",
    "prev_same_source_speed_mps",
    "temporal_continuity_score",
    "prev_state_distance_m",
    "prev_state_dt_s",
    "prev_state_speed_mps",
)


@dataclass(frozen=True)
class ClusterRankerModel:
    """Portable JSON-serializable cluster-ranker model."""

    model_type: str
    feature_columns: list[str]
    feature_means: list[float]
    feature_scales: list[float]
    weights: list[float]
    bias: float
    source_values: list[str]
    constant_score: float | None = None
    sklearn_estimator_base64: str | None = None
    target_column: str = "good_cluster"
    score_transform: str = "probability"
    score_distance_scale_m: float = 10.0


def build_cluster_feature_table(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    truth: pd.DataFrame | None = None,
    good_threshold_m: float = 5.0,
    max_truth_time_delta_s: float = 0.5,
    previous_states: pd.DataFrame | None = None,
    image_evidence: pd.DataFrame | None = None,
    cross_sensor_time_window_s: float = 0.05,
    cross_sensor_distance_gate_m: float = 5.0,
) -> pd.DataFrame:
    """Return candidate cluster features and optional truth-distance labels."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return rows
    rows = _with_default_cluster_geometry(rows)
    rows = _add_frame_rank_features(rows)
    rows = _add_cross_sensor_features(
        rows,
        time_window_s=cross_sensor_time_window_s,
        distance_gate_m=cross_sensor_distance_gate_m,
    )
    rows = _add_temporal_features(rows)
    if previous_states is not None and not previous_states.empty:
        rows = _add_previous_state_features(rows, previous_states)
    if image_evidence is not None and not image_evidence.empty:
        rows = _add_sequence_image_evidence_features(rows, image_evidence)
    if truth is not None and not truth.empty:
        rows = label_cluster_features_against_truth(
            rows,
            truth,
            good_threshold_m=good_threshold_m,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
    return rows.sort_values(["sequence_id", "time_s", "source", "track_id"]).reset_index(drop=True)


def label_cluster_features_against_truth(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    good_threshold_m: float = 5.0,
    max_truth_time_delta_s: float = 0.5,
) -> pd.DataFrame:
    """Attach nearest-truth residuals and good-cluster targets."""

    truth_rows = _truth_rows(truth)
    out = features.copy()
    distances_3d: list[float] = []
    distances_2d: list[float] = []
    vertical_errors: list[float] = []
    time_deltas: list[float] = []
    matched: list[bool] = []
    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=True)
    }
    for _, row in out.iterrows():
        seq_truth = truth_by_sequence.get(str(row["sequence_id"]))
        if seq_truth is None or seq_truth.empty:
            distances_3d.append(np.nan)
            distances_2d.append(np.nan)
            vertical_errors.append(np.nan)
            time_deltas.append(np.nan)
            matched.append(False)
            continue
        truth_t = seq_truth["time_s"].to_numpy(float)
        idx = int(np.argmin(np.abs(truth_t - float(row["time_s"]))))
        dt = float(row["time_s"] - truth_t[idx])
        if abs(dt) > float(max_truth_time_delta_s):
            distances_3d.append(np.nan)
            distances_2d.append(np.nan)
            vertical_errors.append(np.nan)
            time_deltas.append(dt)
            matched.append(False)
            continue
        pred = row[["x_m", "y_m", "z_m"]].to_numpy(float)
        truth_xyz = seq_truth.iloc[idx][["x_m", "y_m", "z_m"]].to_numpy(float)
        residual = pred - truth_xyz
        distances_3d.append(float(np.linalg.norm(residual)))
        distances_2d.append(float(np.linalg.norm(residual[:2])))
        vertical_errors.append(float(abs(residual[2])))
        time_deltas.append(dt)
        matched.append(True)
    out["truth_time_delta_s"] = time_deltas
    out["truth_distance_2d_m"] = distances_2d
    out["truth_distance_3d_m"] = distances_3d
    out["truth_vertical_error_m"] = vertical_errors
    out["truth_matched"] = matched
    out["good_cluster_2m"] = out["truth_distance_3d_m"] <= 2.0
    out["good_cluster_5m"] = out["truth_distance_3d_m"] <= 5.0
    out["good_cluster_10m"] = out["truth_distance_3d_m"] <= 10.0
    out["good_cluster"] = out["truth_distance_3d_m"] <= float(good_threshold_m)
    return out


def train_cluster_ranker(
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
) -> ClusterRankerModel:
    """Train a point-cloud cluster ranker.

    ``model_type="logistic"`` keeps the portable pure-NumPy baseline.  When
    scikit-learn is installed, tree and sklearn-logistic variants can be used
    and are serialized into the same JSON model wrapper.
    """

    model_type = str(model_type)
    actual_target = _actual_target_column(features, model_type=model_type, target_column=target_column)
    rows = features.loc[features[actual_target].notna()].copy()
    if rows.empty:
        raise ValueError(f"no rows with target column {actual_target!r}")
    source_values = sorted(rows["source"].fillna("").astype(str).unique())
    feature_columns = _ranker_feature_columns(rows, source_values)
    matrix = _feature_matrix(rows, feature_columns, source_values=source_values)
    matrix, means, scales = _standardize_training_matrix(matrix)
    x = (matrix - means) / scales
    if model_type != "logistic":
        return _train_sklearn_cluster_ranker(
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
        return ClusterRankerModel(
            model_type="constant-logistic",
            feature_columns=feature_columns,
            feature_means=means.tolist(),
            feature_scales=scales.tolist(),
            weights=[0.0] * len(feature_columns),
            bias=_logit(np.clip(positive_rate, 1.0e-6, 1.0 - 1.0e-6)),
            source_values=source_values,
            constant_score=positive_rate,
            target_column=actual_target,
            score_distance_scale_m=float(score_distance_scale_m),
        )
    weights = np.zeros(x.shape[1], dtype=float)
    bias = _logit(positive_rate)
    for _ in range(max(int(iterations), 1)):
        logits = x @ weights + bias
        pred = _sigmoid(logits)
        error = pred - y
        weights -= float(learning_rate) * ((x.T @ error) / len(y) + float(l2) * weights)
        bias -= float(learning_rate) * float(np.mean(error))
    return ClusterRankerModel(
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


def _actual_target_column(features: pd.DataFrame, *, model_type: str, target_column: str) -> str:
    if model_type.endswith("-regressor") and target_column == "good_cluster":
        if "truth_distance_3d_m" in features.columns:
            return "truth_distance_3d_m"
    return target_column


def _standardize_training_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(matrix)
    means = np.divide(
        np.where(finite_mask, matrix, 0.0).sum(axis=0),
        finite_mask.sum(axis=0),
        out=np.zeros(matrix.shape[1], dtype=float),
        where=finite_mask.sum(axis=0) > 0,
    )
    matrix = np.where(np.isfinite(matrix), matrix, means)
    scales = np.nanstd(matrix, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1.0e-9), scales, 1.0)
    return matrix, means, scales


def _train_sklearn_cluster_ranker(
    x: np.ndarray,
    rows: pd.DataFrame,
    *,
    model_type: str,
    target_column: str,
    feature_columns: list[str],
    feature_means: np.ndarray,
    feature_scales: np.ndarray,
    source_values: list[str],
    random_state: int,
    n_estimators: int,
    score_distance_scale_m: float,
) -> ClusterRankerModel:
    estimator, score_transform = _make_sklearn_estimator(
        model_type=model_type,
        random_state=random_state,
        n_estimators=n_estimators,
    )
    if score_transform == "inverse-distance":
        y = pd.to_numeric(rows[target_column], errors="coerce").to_numpy(float)
    else:
        y = rows[target_column].astype(bool).astype(int).to_numpy()
        unique = np.unique(y)
        if unique.size < 2:
            constant_score = float(unique[0]) if unique.size else 0.0
            return ClusterRankerModel(
                model_type="constant-logistic",
                feature_columns=feature_columns,
                feature_means=feature_means.tolist(),
                feature_scales=feature_scales.tolist(),
                weights=[0.0] * len(feature_columns),
                bias=_logit(np.clip(constant_score, 1.0e-6, 1.0 - 1.0e-6)),
                source_values=source_values,
                constant_score=constant_score,
                target_column=target_column,
                score_distance_scale_m=float(score_distance_scale_m),
            )
    finite = np.isfinite(y)
    if not finite.any():
        raise ValueError(f"no finite target values for {target_column!r}")
    estimator.fit(x[finite], y[finite])
    return ClusterRankerModel(
        model_type=model_type,
        feature_columns=feature_columns,
        feature_means=feature_means.tolist(),
        feature_scales=feature_scales.tolist(),
        weights=[0.0] * len(feature_columns),
        bias=0.0,
        source_values=source_values,
        constant_score=None,
        sklearn_estimator_base64=_encode_sklearn_estimator(estimator),
        target_column=target_column,
        score_transform=score_transform,
        score_distance_scale_m=float(score_distance_scale_m),
    )


def _make_sklearn_estimator(
    *,
    model_type: str,
    random_state: int,
    n_estimators: int,
) -> tuple[Any, str]:
    try:
        if model_type == "sklearn-logistic":
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=1000, class_weight="balanced"), "probability"
        if model_type == "random-forest-classifier":
            from sklearn.ensemble import RandomForestClassifier

            return (
                RandomForestClassifier(
                    n_estimators=max(int(n_estimators), 1),
                    random_state=int(random_state),
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
                "probability",
            )
        if model_type == "hist-gradient-boosting-classifier":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return (
                HistGradientBoostingClassifier(
                    random_state=int(random_state),
                    max_iter=max(int(n_estimators), 1),
                    learning_rate=0.05,
                    l2_regularization=0.01,
                ),
                "probability",
            )
        if model_type == "random-forest-regressor":
            from sklearn.ensemble import RandomForestRegressor

            return (
                RandomForestRegressor(
                    n_estimators=max(int(n_estimators), 1),
                    random_state=int(random_state),
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
                "inverse-distance",
            )
        if model_type == "hist-gradient-boosting-regressor":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return (
                HistGradientBoostingRegressor(
                    random_state=int(random_state),
                    max_iter=max(int(n_estimators), 1),
                    learning_rate=0.05,
                    l2_regularization=0.01,
                    loss="squared_error",
                ),
                "inverse-distance",
            )
    except ImportError as exc:
        raise ValueError(
            f"{model_type} requires scikit-learn; install sklearn or use model_type='logistic'"
        ) from exc
    raise ValueError(f"unsupported cluster ranker model_type={model_type!r}")


def _encode_sklearn_estimator(estimator: Any) -> str:
    return base64.b64encode(pickle.dumps(estimator)).decode("ascii")


def _decode_sklearn_estimator(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


def score_cluster_candidates(
    candidates: CandidateFrame | pd.DataFrame,
    model: ClusterRankerModel,
    *,
    replace_confidence: bool = True,
    previous_states: pd.DataFrame | None = None,
    image_evidence: pd.DataFrame | None = None,
    cross_sensor_time_window_s: float = 0.05,
    cross_sensor_distance_gate_m: float = 5.0,
) -> CandidateFrame:
    """Score cluster candidates and optionally replace ``confidence``."""

    features = build_cluster_feature_table(
        candidates,
        previous_states=previous_states,
        image_evidence=image_evidence,
        cross_sensor_time_window_s=cross_sensor_time_window_s,
        cross_sensor_distance_gate_m=cross_sensor_distance_gate_m,
    )
    if features.empty:
        return CandidateFrame(normalize_candidate_columns(features))
    scores = predict_cluster_scores(features, model)
    rows = features.copy()
    rows["ranker_score"] = scores
    if replace_confidence:
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence", np.nan), errors="coerce")
        rows["confidence"] = scores
    return CandidateFrame(normalize_candidate_columns(rows))


def predict_cluster_scores(features: pd.DataFrame, model: ClusterRankerModel) -> np.ndarray:
    """Predict good-cluster probabilities for feature rows."""

    if features.empty:
        return np.asarray([], dtype=float)
    matrix = _feature_matrix(features, model.feature_columns, source_values=model.source_values)
    means = np.asarray(model.feature_means, dtype=float)
    scales = np.asarray(model.feature_scales, dtype=float)
    matrix = np.where(np.isfinite(matrix), matrix, means)
    x = (matrix - means) / scales
    if model.sklearn_estimator_base64:
        estimator = _decode_sklearn_estimator(model.sklearn_estimator_base64)
        if model.score_transform == "inverse-distance":
            distances = np.asarray(estimator.predict(x), dtype=float)
            distances = np.maximum(np.nan_to_num(distances, nan=1.0e6), 0.0)
            scale = max(float(model.score_distance_scale_m), 1.0e-6)
            return 1.0 / (1.0 + distances / scale)
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(x)
            if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                return np.asarray(probabilities[:, 1], dtype=float)
            return np.asarray(probabilities).reshape(-1).astype(float)
        if hasattr(estimator, "decision_function"):
            return _sigmoid(np.asarray(estimator.decision_function(x), dtype=float))
        return np.asarray(estimator.predict(x), dtype=float)
    if model.constant_score is not None:
        return np.full(len(features), float(model.constant_score), dtype=float)
    logits = x @ np.asarray(model.weights, dtype=float) + float(model.bias)
    return _sigmoid(logits)


def merge_cross_sensor_candidate_clusters(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    time_window_s: float = 0.05,
    distance_gate_m: float = 5.0,
) -> CandidateFrame:
    """Create extra cross-sensor merged candidates for nearby same-time clusters."""

    rows = _candidate_rows(candidates)
    merged: list[dict[str, Any]] = []
    emitted_components: set[tuple[str, tuple[int, ...]]] = set()
    for sequence_id, seq_rows in rows.groupby("sequence_id", sort=True):
        seq_rows = (
            seq_rows.assign(_candidate_row_id=np.arange(len(seq_rows), dtype=int))
            .sort_values("time_s")
            .reset_index(drop=True)
        )
        for time_s, group in seq_rows.groupby("time_s", sort=True):
            nearby = seq_rows.loc[
                np.abs(seq_rows["time_s"].to_numpy(float) - float(time_s))
                <= float(time_window_s)
            ].reset_index(drop=True)
            if len(nearby) < 2 or nearby["source"].astype(str).nunique() < 2:
                continue
            xyz = nearby[["x_m", "y_m", "z_m"]].to_numpy(float)
            components = _distance_components(xyz, max_distance_m=distance_gate_m)
            for component_index, indices in enumerate(components):
                comp = nearby.iloc[indices].copy()
                if len(comp) < 2 or comp["source"].astype(str).nunique() < 2:
                    continue
                component_key = (
                    str(sequence_id),
                    tuple(sorted(int(row_id) for row_id in comp["_candidate_row_id"])),
                )
                if component_key in emitted_components:
                    continue
                emitted_components.add(component_key)
                weights = _numeric_series(comp, "confidence", default=1.0)
                weights = weights.fillna(1.0).clip(lower=1.0e-6).to_numpy(float)
                centroid = np.average(comp[["x_m", "y_m", "z_m"]].to_numpy(float), axis=0, weights=weights)
                extent = np.ptp(comp[["x_m", "y_m", "z_m"]].to_numpy(float), axis=0)
                merged.append(
                    {
                        "sequence_id": str(sequence_id),
                        "time_s": float(np.average(comp["time_s"].to_numpy(float), weights=weights)),
                        "source": "cross-sensor-merged",
                        "track_id": (
                            f"cross-sensor:{sequence_id}:{float(time_s):.6f}:{component_index}"
                        ),
                        "x_m": float(centroid[0]),
                        "y_m": float(centroid[1]),
                        "z_m": float(centroid[2]),
                        "std_xy_m": float(max(np.linalg.norm(extent[:2]), 0.5)),
                        "std_z_m": float(max(extent[2], 0.5)),
                        "confidence": float(np.sum(weights)),
                        "class_name": "uav",
                        "cluster_point_count": int(
                            _numeric_series(comp, "cluster_point_count", default=1.0).fillna(1.0).sum()
                        ),
                        "cluster_extent_x_m": float(extent[0]),
                        "cluster_extent_y_m": float(extent[1]),
                        "cluster_extent_z_m": float(extent[2]),
                        "cross_sensor_neighbor_count": int(len(comp)),
                    }
                )
    if not merged:
        return CandidateFrame(normalize_candidate_columns(rows.iloc[0:0].copy()))
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(merged)))


def save_cluster_ranker_model(model: ClusterRankerModel, path: Path) -> Path:
    """Write a cluster-ranker model JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.__dict__, indent=2), encoding="utf-8")
    return path


def load_cluster_ranker_model(path: Path) -> ClusterRankerModel:
    """Read a cluster-ranker model JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ClusterRankerModel(
        model_type=str(payload["model_type"]),
        feature_columns=[str(item) for item in payload["feature_columns"]],
        feature_means=[float(item) for item in payload["feature_means"]],
        feature_scales=[float(item) for item in payload["feature_scales"]],
        weights=[float(item) for item in payload["weights"]],
        bias=float(payload["bias"]),
        source_values=[str(item) for item in payload.get("source_values", [])],
        constant_score=(
            None
            if payload.get("constant_score") is None
            else float(payload["constant_score"])
        ),
        sklearn_estimator_base64=payload.get("sklearn_estimator_base64"),
        target_column=str(payload.get("target_column", "good_cluster")),
        score_transform=str(payload.get("score_transform", "probability")),
        score_distance_scale_m=float(payload.get("score_distance_scale_m", 10.0)),
    )


def write_ranker_diagnostics(features: pd.DataFrame, path: Path) -> Path:
    """Write feature/label diagnostics for inspection."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return path


def evaluate_cluster_ranker_loso(
    features: pd.DataFrame,
    *,
    model_type: str = "logistic",
    target_column: str = "good_cluster",
    learning_rate: float = 0.05,
    iterations: int = 600,
    random_state: int = 13,
    n_estimators: int = 200,
    score_distance_scale_m: float = 10.0,
    min_train_sequences: int = 1,
    protocol: str = "LOSO public-validation diagnostic, not submission-valid",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a cluster ranker with leave-one-sequence-out folds."""

    rows = pd.DataFrame(features).copy()
    if rows.empty:
        raise ValueError("no feature rows for LOSO cluster-ranker evaluation")
    if "sequence_id" not in rows.columns:
        raise ValueError("LOSO cluster-ranker evaluation requires sequence_id")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    sequences = sorted(rows["sequence_id"].dropna().unique())
    if len(sequences) < 2:
        raise ValueError("LOSO cluster-ranker evaluation requires at least two sequences")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for heldout_sequence in sequences:
        train_rows = rows.loc[rows["sequence_id"] != heldout_sequence].copy()
        heldout_rows = rows.loc[rows["sequence_id"] == heldout_sequence].copy()
        train_sequences = sorted(train_rows["sequence_id"].dropna().unique())
        if len(train_sequences) < int(min_train_sequences) or train_rows.empty or heldout_rows.empty:
            continue
        model = train_cluster_ranker(
            train_rows,
            model_type=model_type,
            target_column=target_column,
            learning_rate=learning_rate,
            iterations=iterations,
            random_state=random_state,
            n_estimators=n_estimators,
            score_distance_scale_m=score_distance_scale_m,
        )
        heldout_rows["ranker_score"] = predict_cluster_scores(heldout_rows, model)
        heldout_rows["raw_confidence"] = pd.to_numeric(
            heldout_rows.get("confidence", np.nan),
            errors="coerce",
        )
        heldout_rows["confidence"] = heldout_rows["ranker_score"]
        heldout_rows["loso_heldout_sequence"] = heldout_sequence
        heldout_rows["loso_train_sequence_count"] = int(len(train_sequences))
        heldout_rows["loso_model_type"] = model.model_type
        heldout_rows["loso_target_column"] = model.target_column
        heldout_rows["loso_protocol"] = protocol
        predictions.append(heldout_rows)
        fold_summary = _ranker_prediction_summary(
            heldout_rows,
            sequence=heldout_sequence,
            split="heldout_sequence",
            protocol=protocol,
        )
        fold_summary["train_sequence_count"] = int(len(train_sequences))
        fold_summary["model_type"] = model.model_type
        fold_summary["target_column"] = model.target_column
        fold_rows.append(fold_summary)
    if not predictions:
        raise ValueError("no LOSO folds could be evaluated")
    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_summary_frame = pd.DataFrame.from_records(fold_rows).sort_values("sequence_id")
    pooled_summary = pd.DataFrame.from_records(
        [
            _ranker_prediction_summary(
                prediction_frame,
                sequence="__pooled__",
                split="pooled_loso",
                protocol=protocol,
            )
            | {
                "fold_count": int(len(fold_summary_frame)),
                "sequence_count": int(len(sequences)),
                "model_type": str(model_type),
                "target_column": str(target_column),
            }
        ]
    )
    return prediction_frame, fold_summary_frame.reset_index(drop=True), pooled_summary


def _ranker_prediction_summary(
    rows: pd.DataFrame,
    *,
    sequence: str,
    split: str,
    protocol: str,
) -> dict[str, Any]:
    labeled = rows.loc[pd.to_numeric(rows.get("truth_distance_3d_m"), errors="coerce").notna()].copy()
    frame_rows = _ranker_frame_selection_rows(labeled)
    selected = pd.to_numeric(frame_rows.get("selected_truth_distance_3d_m"), errors="coerce")
    oracle = pd.to_numeric(frame_rows.get("oracle_truth_distance_3d_m"), errors="coerce")
    regret = pd.to_numeric(frame_rows.get("candidate_regret_3d_m"), errors="coerce")
    good = rows.get("good_cluster")
    good_values = pd.Series(good, index=rows.index).fillna(False).astype(bool) if good is not None else pd.Series(dtype=bool)
    return {
        "protocol": protocol,
        "split": split,
        "sequence_id": sequence,
        "candidate_rows": int(len(rows)),
        "labeled_candidate_rows": int(len(labeled)),
        "positive_candidate_rows": int(good_values.sum()),
        "positive_candidate_rate": _safe_mean(good_values.astype(float)),
        "frame_count": int(len(frame_rows)),
        "top1_mean_3d_m": _safe_mean(selected),
        "top1_median_3d_m": _safe_quantile(selected, 0.50),
        "top1_p95_3d_m": _safe_quantile(selected, 0.95),
        "top1_max_3d_m": _safe_max(selected),
        "oracle_mean_3d_m": _safe_mean(oracle),
        "oracle_p95_3d_m": _safe_quantile(oracle, 0.95),
        "candidate_regret_mean_3d_m": _safe_mean(regret),
        "candidate_regret_p95_3d_m": _safe_quantile(regret, 0.95),
        "top1_within_2m": _within_rate(selected, 2.0),
        "top1_within_5m": _within_rate(selected, 5.0),
        "top1_within_10m": _within_rate(selected, 10.0),
        "oracle_within_2m": _within_rate(oracle, 2.0),
        "oracle_within_5m": _within_rate(oracle, 5.0),
        "oracle_within_10m": _within_rate(oracle, 10.0),
        "score_auc": _binary_auc(
            pd.Series(rows.get("ranker_score"), index=rows.index),
            good_values,
        ),
    }


def _ranker_frame_selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "time_s",
                "selected_truth_distance_3d_m",
                "oracle_truth_distance_3d_m",
                "candidate_regret_3d_m",
            ]
        )
    out: list[dict[str, Any]] = []
    work = rows.copy()
    work["truth_distance_3d_m"] = pd.to_numeric(work["truth_distance_3d_m"], errors="coerce")
    work["ranker_score"] = pd.to_numeric(work.get("ranker_score"), errors="coerce")
    for (sequence_id, time_s), group in work.groupby(["sequence_id", "time_s"], sort=True):
        group = group.loc[group["truth_distance_3d_m"].notna()].copy()
        if group.empty:
            continue
        score = group["ranker_score"].fillna(float("-inf"))
        selected = group.iloc[int(np.argmax(score.to_numpy(float)))]
        oracle_distance = float(group["truth_distance_3d_m"].min())
        selected_distance = float(selected["truth_distance_3d_m"])
        out.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "selected_source": str(selected.get("source", "")),
                "selected_track_id": str(selected.get("track_id", "")),
                "selected_ranker_score": float(selected.get("ranker_score", np.nan)),
                "selected_truth_distance_3d_m": selected_distance,
                "oracle_truth_distance_3d_m": oracle_distance,
                "candidate_regret_3d_m": selected_distance - oracle_distance,
                "candidate_count": int(len(group)),
            }
        )
    return pd.DataFrame.from_records(out)


def _safe_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.quantile(float(quantile))) if len(finite) else float("nan")


def _safe_max(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.max()) if len(finite) else float("nan")


def _within_rate(values: pd.Series, threshold_m: float) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float((finite <= float(threshold_m)).mean()) if len(finite) else float("nan")


def _binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    score_values = pd.to_numeric(scores, errors="coerce")
    label_values = labels.fillna(False).astype(bool)
    valid = score_values.notna() & label_values.notna()
    if not valid.any():
        return float("nan")
    score_array = score_values.loc[valid].to_numpy(float)
    label_array = label_values.loc[valid].to_numpy(bool)
    positives = int(label_array.sum())
    negatives = int((~label_array).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score_array, kind="mergesort")
    ranks = np.empty(len(score_array), dtype=float)
    sorted_scores = score_array[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    positive_rank_sum = float(ranks[label_array].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates).copy()
    if rows.empty:
        return normalize_candidate_columns(rows)
    return normalize_candidate_columns(rows)


def _truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    if {"sequence_id", "time_s", "x_m", "y_m", "z_m"}.issubset(truth.columns):
        rows = truth[["sequence_id", "time_s", "x_m", "y_m", "z_m"]].copy()
    else:
        rows = load_evaluation_truth_file(Path(truth)).rows if isinstance(truth, Path) else pd.DataFrame(truth)
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.loc[np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def _with_default_cluster_geometry(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "cluster_point_count" not in out.columns:
        out["cluster_point_count"] = _numeric_series(out, "confidence", default=1.0)
    for column in (
        "cluster_extent_x_m",
        "cluster_extent_y_m",
        "cluster_extent_z_m",
        "cluster_extent_xy_m",
        "cluster_extent_3d_m",
        "cluster_bbox_volume_m3",
        "cluster_density_points_per_m3",
        "cluster_range_xy_m",
        "cluster_range_3d_m",
        "cluster_height_m",
    ):
        if column not in out.columns:
            out[column] = np.nan
    xyz = out[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    range_xy_fallback = pd.Series(np.hypot(xyz["x_m"], xyz["y_m"]), index=out.index)
    range_3d_fallback = pd.Series(np.linalg.norm(xyz.to_numpy(float), axis=1), index=out.index)
    out["cluster_range_xy_m"] = _numeric_series(out, "cluster_range_xy_m").fillna(range_xy_fallback)
    out["cluster_range_3d_m"] = _numeric_series(out, "cluster_range_3d_m").fillna(range_3d_fallback)
    out["cluster_height_m"] = _numeric_series(out, "cluster_height_m").fillna(xyz["z_m"])
    return out


def _add_frame_rank_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if out.empty:
        return out
    out["frame_candidate_count"] = 0
    out["frame_source_candidate_count"] = 0
    for _, group in out.groupby(["sequence_id", "time_s"], sort=False):
        frame_indices = group.index
        out.loc[frame_indices, "frame_candidate_count"] = int(len(group))
        out.loc[frame_indices, "frame_rank_confidence_desc"] = _rank_desc(
            _numeric_series(group, "confidence", default=0.0)
        ).to_numpy()
        out.loc[frame_indices, "frame_rank_point_count_desc"] = _rank_desc(
            _numeric_series(group, "cluster_point_count", default=0.0)
        ).to_numpy()
        out.loc[frame_indices, "frame_rank_density_desc"] = _rank_desc(
            _numeric_series(group, "cluster_density_points_per_m3", default=0.0)
        ).to_numpy()
        out.loc[frame_indices, "frame_rank_range_3d_asc"] = _rank_asc(
            _numeric_series(group, "cluster_range_3d_m", default=np.inf)
        ).to_numpy()
        out.loc[frame_indices, "frame_rank_height_abs_asc"] = _rank_asc(
            _numeric_series(group, "cluster_height_m", default=np.inf).abs()
        ).to_numpy()
    for _, group in out.groupby(["sequence_id", "time_s", "source"], sort=False):
        source_indices = group.index
        out.loc[source_indices, "frame_source_candidate_count"] = int(len(group))
        out.loc[source_indices, "source_frame_rank_confidence_desc"] = _rank_desc(
            _numeric_series(group, "confidence", default=0.0)
        ).to_numpy()
        out.loc[source_indices, "source_frame_rank_point_count_desc"] = _rank_desc(
            _numeric_series(group, "cluster_point_count", default=0.0)
        ).to_numpy()
        out.loc[source_indices, "source_frame_rank_density_desc"] = _rank_desc(
            _numeric_series(group, "cluster_density_points_per_m3", default=0.0)
        ).to_numpy()
        out.loc[source_indices, "source_frame_rank_range_3d_asc"] = _rank_asc(
            _numeric_series(group, "cluster_range_3d_m", default=np.inf)
        ).to_numpy()
    out["frame_source_fraction"] = (
        pd.to_numeric(out["frame_source_candidate_count"], errors="coerce")
        / pd.to_numeric(out["frame_candidate_count"], errors="coerce").replace(0, np.nan)
    )
    return out


def _rank_desc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(-np.inf)
    return numeric.rank(method="min", ascending=False)


def _rank_asc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(np.inf)
    return numeric.rank(method="min", ascending=True)


def _add_cross_sensor_features(
    rows: pd.DataFrame,
    *,
    time_window_s: float,
    distance_gate_m: float,
) -> pd.DataFrame:
    out = rows.copy()
    distances = np.full(len(out), np.nan, dtype=float)
    counts = np.zeros(len(out), dtype=int)
    for _, seq_rows in out.groupby("sequence_id", sort=False):
        seq_indices = seq_rows.index.to_numpy()
        t = seq_rows["time_s"].to_numpy(float)
        xyz = seq_rows[["x_m", "y_m", "z_m"]].to_numpy(float)
        sources = seq_rows["source"].astype(str).to_numpy()
        for local_idx, global_idx in enumerate(seq_indices):
            mask = (np.abs(t - t[local_idx]) <= float(time_window_s)) & (
                sources != sources[local_idx]
            )
            if not mask.any():
                continue
            d = np.linalg.norm(xyz[mask] - xyz[local_idx], axis=1)
            distances[global_idx] = float(np.min(d))
            counts[global_idx] = int(np.sum(d <= float(distance_gate_m)))
    out["nearest_cross_sensor_distance_m"] = distances
    out["nearest_cross_sensor_score"] = 1.0 / (1.0 + np.nan_to_num(distances, nan=1.0e6))
    out["cross_sensor_neighbor_count"] = counts
    return out


def _add_temporal_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["prev_same_source_distance_m"] = np.nan
    out["prev_same_source_dt_s"] = np.nan
    out["prev_same_source_speed_mps"] = np.nan
    for _, group in out.groupby(["sequence_id", "source"], sort=False):
        group = group.sort_values("time_s")
        prev_xyz: np.ndarray | None = None
        prev_time: float | None = None
        for index, row in group.iterrows():
            xyz = row[["x_m", "y_m", "z_m"]].to_numpy(float)
            time_s = float(row["time_s"])
            if prev_xyz is not None and prev_time is not None:
                distance = float(np.linalg.norm(xyz - prev_xyz))
                dt = max(time_s - prev_time, 1.0e-6)
                out.loc[index, "prev_same_source_distance_m"] = distance
                out.loc[index, "prev_same_source_dt_s"] = dt
                out.loc[index, "prev_same_source_speed_mps"] = distance / dt
            prev_xyz = xyz
            prev_time = time_s
    out["temporal_continuity_score"] = 1.0 / (
        1.0 + pd.to_numeric(out["prev_same_source_distance_m"], errors="coerce").fillna(1.0e6)
    )
    return out


def _add_previous_state_features(rows: pd.DataFrame, previous_states: pd.DataFrame) -> pd.DataFrame:
    states = _state_rows(previous_states)
    out = rows.copy()
    out["prev_state_distance_m"] = np.nan
    out["prev_state_dt_s"] = np.nan
    out["prev_state_speed_mps"] = np.nan
    state_by_sequence = {
        str(sequence_id): group.sort_values("time_s")
        for sequence_id, group in states.groupby("sequence_id", sort=True)
    }
    for index, row in out.iterrows():
        group = state_by_sequence.get(str(row["sequence_id"]))
        if group is None or group.empty:
            continue
        prior = group.loc[group["time_s"] <= float(row["time_s"])]
        if prior.empty:
            continue
        state = prior.iloc[-1]
        dt = max(float(row["time_s"]) - float(state["time_s"]), 1.0e-6)
        distance = float(
            np.linalg.norm(
                row[["x_m", "y_m", "z_m"]].to_numpy(float)
                - state[["x_m", "y_m", "z_m"]].to_numpy(float)
            )
        )
        out.loc[index, "prev_state_distance_m"] = distance
        out.loc[index, "prev_state_dt_s"] = dt
        out.loc[index, "prev_state_speed_mps"] = distance / dt
    return out


def _state_rows(states: pd.DataFrame) -> pd.DataFrame:
    rows = states.copy()
    rename = {}
    for source, target in {
        "state_x_m": "x_m",
        "state_y_m": "y_m",
        "state_z_m": "z_m",
        "timestamp": "time_s",
    }.items():
        if source in rows.columns and target not in rows.columns:
            rename[source] = target
    rows = rows.rename(columns=rename)
    required = ["sequence_id", "time_s", "x_m", "y_m", "z_m"]
    missing = set(required).difference(rows.columns)
    if missing:
        raise ValueError(f"previous-state rows missing columns: {sorted(missing)}")
    rows = rows[required].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows.loc[np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def _add_sequence_image_evidence_features(
    rows: pd.DataFrame,
    image_evidence: pd.DataFrame,
) -> pd.DataFrame:
    evidence = _sequence_image_evidence_rows(image_evidence)
    if evidence.empty:
        return rows
    out = rows.merge(evidence, on="sequence_id", how="left")
    out["image_evidence_available"] = out["image_evidence_available"].fillna(0.0)
    return out


def _sequence_image_evidence_rows(image_evidence: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(image_evidence).copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return pd.DataFrame(columns=["sequence_id", "image_evidence_available"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    keep = ["sequence_id"]
    for column in rows.columns:
        text = str(column)
        if not text.startswith("image_"):
            continue
        numeric = pd.to_numeric(rows[column], errors="coerce")
        if numeric.notna().any():
            rows[text] = numeric
            keep.append(text)
    evidence = rows[keep].groupby("sequence_id", as_index=False).mean(numeric_only=True)
    evidence["image_evidence_available"] = 1.0
    return evidence


def _ranker_feature_columns(rows: pd.DataFrame, source_values: list[str]) -> list[str]:
    columns = [column for column in BASE_CLUSTER_FEATURE_COLUMNS if column in rows.columns]
    columns.extend(_image_evidence_feature_columns(rows))
    columns.extend(f"source={source}" for source in source_values)
    return columns


def _image_evidence_feature_columns(rows: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in rows.columns:
        text = str(column)
        if not text.startswith("image_"):
            continue
        if pd.to_numeric(rows[column], errors="coerce").notna().any():
            columns.append(text)
    return sorted(set(columns))


def _feature_matrix(
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    source_values: list[str],
) -> np.ndarray:
    source_text = rows.get("source", pd.Series([""] * len(rows), index=rows.index)).fillna("").astype(str)
    matrix_columns: dict[str, pd.Series] = {}
    for column in feature_columns:
        if column.startswith("source="):
            source = column.split("=", 1)[1]
            matrix_columns[column] = (source_text == source).astype(float)
        else:
            matrix_columns[column] = _numeric_series(rows, column)
    return pd.DataFrame(matrix_columns, index=rows.index).to_numpy(float)


def _numeric_series(rows: pd.DataFrame, column: str, *, default: float = np.nan) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _distance_components(xyz: np.ndarray, *, max_distance_m: float) -> list[list[int]]:
    n = int(len(xyz))
    seen = np.zeros(n, dtype=bool)
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            idx = stack.pop()
            component.append(idx)
            distances = np.linalg.norm(xyz - xyz[idx], axis=1)
            for candidate in np.flatnonzero((distances <= float(max_distance_m)) & ~seen):
                seen[int(candidate)] = True
                stack.append(int(candidate))
        components.append(component)
    return components


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _logit(value: float) -> float:
    value = float(np.clip(value, 1.0e-6, 1.0 - 1.0e-6))
    return float(np.log(value / (1.0 - value)))


def _load_candidates(path: Path) -> CandidateFrame:
    return CandidateFrame(normalize_candidate_columns(pd.read_csv(path)))


def _load_sequence_root_candidates(
    root: Path,
    *,
    sequence_glob: str,
    apply_calibration: bool,
    voxel_size_m: float,
    min_cluster_points: int,
) -> CandidateFrame:
    frames: list[CandidateFrame] = []
    sequences = discover_sequence_paths(Path(root), sequence_glob=sequence_glob)
    for paths in sequences:
        try:
            candidates, _, _ = load_sequence_export(
                paths,
                apply_calibration=apply_calibration,
                voxel_size_m=voxel_size_m,
                min_cluster_points=min_cluster_points,
            )
        except Exception:
            continue
        if not candidates.rows.empty:
            frames.append(candidates)
    if not frames:
        raise ValueError(f"no candidate rows loaded from sequence root {root}")
    return merge_candidate_frames(frames)


def _load_candidates_from_args(
    *,
    csv_path: Path | None,
    sequence_root: Path | None,
    sequence_glob: str,
    apply_calibration: bool,
    voxel_size_m: float,
    min_cluster_points: int,
) -> CandidateFrame:
    frames: list[CandidateFrame] = []
    if csv_path is not None:
        frames.append(_load_candidates(csv_path))
    if sequence_root is not None:
        frames.append(
            _load_sequence_root_candidates(
                sequence_root,
                sequence_glob=sequence_glob,
                apply_calibration=apply_calibration,
                voxel_size_m=voxel_size_m,
                min_cluster_points=min_cluster_points,
            )
        )
    if not frames:
        raise ValueError("provide a candidate CSV or sequence root")
    return merge_candidate_frames(frames)


def _load_truth(path: Path | None) -> pd.DataFrame | None:
    return None if path is None else load_evaluation_truth_file(Path(path)).rows


def _load_states(path: Path | None) -> pd.DataFrame | None:
    return None if path is None else pd.read_csv(path)


def _load_image_evidence(path: Path | None) -> pd.DataFrame | None:
    return None if path is None else pd.read_csv(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-cluster-ranker",
        description="train or apply a supervised point-cloud cluster ranker",
    )
    parser.add_argument("--train-candidates", type=Path, help="training candidate CSV")
    parser.add_argument("--train-sequence-root", type=Path, help="training MMUAD sequence root")
    parser.add_argument("--train-truth", type=Path, help="training truth CSV/ZIP")
    parser.add_argument("--score-candidates", type=Path, help="candidate CSV to score")
    parser.add_argument("--score-sequence-root", type=Path, help="MMUAD sequence root to score")
    parser.add_argument("--previous-states", type=Path, help="optional state/estimate CSV")
    parser.add_argument("--train-image-evidence-csv", type=Path)
    parser.add_argument("--score-image-evidence-csv", type=Path)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=(
            "logistic",
            "sklearn-logistic",
            "random-forest-classifier",
            "hist-gradient-boosting-classifier",
            "random-forest-regressor",
            "hist-gradient-boosting-regressor",
        ),
        default="logistic",
    )
    parser.add_argument("--target-column", default="good_cluster")
    parser.add_argument("--train-features-csv", type=Path)
    parser.add_argument("--score-features-csv", type=Path)
    parser.add_argument("--train-candidates-output-csv", type=Path)
    parser.add_argument("--score-candidates-output-csv", type=Path)
    parser.add_argument("--scored-candidates-csv", type=Path)
    parser.add_argument("--merged-candidates-csv", type=Path)
    parser.add_argument(
        "--loso-eval",
        action="store_true",
        help="run leave-one-sequence-out candidate-ranker evaluation on the training input",
    )
    parser.add_argument("--loso-predictions-csv", type=Path)
    parser.add_argument("--loso-fold-summary-csv", type=Path)
    parser.add_argument("--loso-summary-csv", type=Path)
    parser.add_argument("--loso-protocol-json", type=Path)
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--cross-sensor-time-window-s", type=float, default=0.05)
    parser.add_argument("--cross-sensor-distance-gate-m", type=float, default=5.0)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--score-sequence-glob")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--score-distance-scale-m", type=float)
    args = parser.parse_args(argv)

    if args.train_candidates is not None or args.train_sequence_root is not None:
        if args.train_truth is None:
            raise SystemExit("--train-candidates/--train-sequence-root requires --train-truth")
        train_candidates = _load_candidates_from_args(
            csv_path=args.train_candidates,
            sequence_root=args.train_sequence_root,
            sequence_glob=args.sequence_glob,
            apply_calibration=not args.no_apply_calibration,
            voxel_size_m=args.voxel_size_m,
            min_cluster_points=args.min_cluster_points,
        )
        if args.train_candidates_output_csv is not None:
            args.train_candidates_output_csv.parent.mkdir(parents=True, exist_ok=True)
            train_candidates.rows.to_csv(args.train_candidates_output_csv, index=False)
        train_features = build_cluster_feature_table(
            train_candidates,
            truth=_load_truth(args.train_truth),
            good_threshold_m=args.good_threshold_m,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
            previous_states=_load_states(args.previous_states),
            image_evidence=_load_image_evidence(args.train_image_evidence_csv),
            cross_sensor_time_window_s=args.cross_sensor_time_window_s,
            cross_sensor_distance_gate_m=args.cross_sensor_distance_gate_m,
        )
        model = train_cluster_ranker(
            train_features,
            model_type=args.model_type,
            target_column=args.target_column,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            score_distance_scale_m=(
                args.score_distance_scale_m
                if args.score_distance_scale_m is not None
                else args.good_threshold_m
            ),
        )
        save_cluster_ranker_model(model, args.model_json)
        if args.train_features_csv is not None:
            write_ranker_diagnostics(train_features, args.train_features_csv)
        if args.loso_eval:
            loso_predictions, loso_fold_summary, loso_summary = evaluate_cluster_ranker_loso(
                train_features,
                model_type=args.model_type,
                target_column=args.target_column,
                learning_rate=args.learning_rate,
                iterations=args.iterations,
                random_state=args.random_state,
                n_estimators=args.n_estimators,
                score_distance_scale_m=(
                    args.score_distance_scale_m
                    if args.score_distance_scale_m is not None
                    else args.good_threshold_m
                ),
            )
            if args.loso_predictions_csv is not None:
                args.loso_predictions_csv.parent.mkdir(parents=True, exist_ok=True)
                loso_predictions.to_csv(args.loso_predictions_csv, index=False)
            if args.loso_fold_summary_csv is not None:
                args.loso_fold_summary_csv.parent.mkdir(parents=True, exist_ok=True)
                loso_fold_summary.to_csv(args.loso_fold_summary_csv, index=False)
            if args.loso_summary_csv is not None:
                args.loso_summary_csv.parent.mkdir(parents=True, exist_ok=True)
                loso_summary.to_csv(args.loso_summary_csv, index=False)
            if args.loso_protocol_json is not None:
                args.loso_protocol_json.parent.mkdir(parents=True, exist_ok=True)
                protocol_payload = {
                    "protocol": "LOSO public-validation diagnostic, not submission-valid",
                    "model_type": args.model_type,
                    "target_column": args.target_column,
                    "good_threshold_m": float(args.good_threshold_m),
                    "max_truth_time_delta_s": float(args.max_truth_time_delta_s),
                    "sequence_count": int(loso_summary.loc[0, "sequence_count"]),
                    "fold_count": int(loso_summary.loc[0, "fold_count"]),
                    "candidate_rows": int(loso_summary.loc[0, "candidate_rows"]),
                    "frame_count": int(loso_summary.loc[0, "frame_count"]),
                }
                args.loso_protocol_json.write_text(
                    json.dumps(protocol_payload, indent=2),
                    encoding="utf-8",
                )
            print("cluster_ranker_loso=ok")
            print(f"loso_folds={int(loso_summary.loc[0, 'fold_count'])}")
            print(f"loso_top1_mean_3d_m={loso_summary.loc[0, 'top1_mean_3d_m']}")
            print(f"loso_candidate_regret_p95_3d_m={loso_summary.loc[0, 'candidate_regret_p95_3d_m']}")
        print("cluster_ranker_train=ok")
        print(f"model_json={args.model_json}")
        print(f"model_type={model.model_type}")
        print(f"train_rows={len(train_features)}")
        print(f"positive_rows={int(train_features['good_cluster'].fillna(False).sum())}")
    else:
        model = load_cluster_ranker_model(args.model_json)

    if args.score_candidates is not None or args.score_sequence_root is not None:
        candidates = _load_candidates_from_args(
            csv_path=args.score_candidates,
            sequence_root=args.score_sequence_root,
            sequence_glob=args.score_sequence_glob or args.sequence_glob,
            apply_calibration=not args.no_apply_calibration,
            voxel_size_m=args.voxel_size_m,
            min_cluster_points=args.min_cluster_points,
        )
        if args.score_candidates_output_csv is not None:
            args.score_candidates_output_csv.parent.mkdir(parents=True, exist_ok=True)
            candidates.rows.to_csv(args.score_candidates_output_csv, index=False)
        merged = merge_cross_sensor_candidate_clusters(
            candidates,
            time_window_s=args.cross_sensor_time_window_s,
            distance_gate_m=args.cross_sensor_distance_gate_m,
        )
        if args.merged_candidates_csv is not None:
            args.merged_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
            merged.rows.to_csv(args.merged_candidates_csv, index=False)
        score_input = merge_candidate_frames([candidates, merged])
        scored = score_cluster_candidates(
            score_input,
            load_cluster_ranker_model(args.model_json),
            previous_states=_load_states(args.previous_states),
            image_evidence=_load_image_evidence(args.score_image_evidence_csv),
            cross_sensor_time_window_s=args.cross_sensor_time_window_s,
            cross_sensor_distance_gate_m=args.cross_sensor_distance_gate_m,
        )
        if args.scored_candidates_csv is None:
            raise SystemExit("--score-candidates requires --scored-candidates-csv")
        args.scored_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.rows.to_csv(args.scored_candidates_csv, index=False)
        if args.score_features_csv is not None:
            features = build_cluster_feature_table(
                score_input,
                previous_states=_load_states(args.previous_states),
                image_evidence=_load_image_evidence(args.score_image_evidence_csv),
                cross_sensor_time_window_s=args.cross_sensor_time_window_s,
                cross_sensor_distance_gate_m=args.cross_sensor_distance_gate_m,
            )
            features["ranker_score"] = predict_cluster_scores(features, load_cluster_ranker_model(args.model_json))
            write_ranker_diagnostics(features, args.score_features_csv)
        print("cluster_ranker_score=ok")
        print(f"scored_candidates_csv={args.scored_candidates_csv}")
        print(f"scored_rows={len(scored.rows)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
