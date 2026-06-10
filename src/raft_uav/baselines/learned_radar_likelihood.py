"""Learned radar association likelihoods.

The model is intentionally dependency-light: a standardized logistic likelihood
trained on truth-labeled radar candidates. It predicts the probability that a
candidate row is the UAV and can be used to replace hand-tuned association
scores while leaving the Kalman update unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MODEL_TYPE = "raft-uav.learned-radar-association-logistic-v1"
STATEFUL_COST_METADATA_KEY = "stateful_transition_costs"


DEFAULT_FEATURE_NAMES: tuple[str, ...] = (
    "log1p_association_nis",
    "sqrt_association_nis",
    "position_residual_norm_m",
    "horizontal_residual_norm_m",
    "abs_vertical_residual_m",
    "candidate_count",
    "candidate_rank_by_nis",
    "cat_prob_uav",
    "neg_log_cat_prob_uav",
    "missing_cat_prob_uav",
    "track_id_changed",
    "same_track_id",
    "missing_track_id",
    "velocity_residual_norm_mps",
    "speed_mps",
    "missing_velocity",
    "log1p_range_m",
    "abs_radial_velocity_mps",
    "missing_radial_velocity",
    "log1p_num_inliers",
    "missing_num_inliers",
)


@dataclass(frozen=True)
class LearnedRadarAssociationModel:
    """Standardized logistic model for radar association."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        scale = np.asarray(self.scale, dtype=float).reshape(-1)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        intercept = float(self.intercept)
        if not names:
            raise ValueError("feature_names must not be empty")
        if mean.size != len(names) or scale.size != len(names) or weights.size != len(names):
            raise ValueError("mean, scale, and weights must match feature_names")
        if not np.isfinite(mean).all():
            raise ValueError("mean must be finite")
        if not np.isfinite(weights).all():
            raise ValueError("weights must be finite")
        if not math.isfinite(intercept):
            raise ValueError("intercept must be finite")
        scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "intercept", intercept)

    def logits_from_features(self, features: pd.DataFrame) -> np.ndarray:
        matrix = _raw_feature_matrix(features, self.feature_names)
        filled = np.where(np.isfinite(matrix), matrix, self.mean.reshape(1, -1))
        standardized = (filled - self.mean.reshape(1, -1)) / self.scale.reshape(1, -1)
        return standardized @ self.weights + self.intercept

    def predict_proba_features(self, features: pd.DataFrame) -> np.ndarray:
        return _sigmoid(self.logits_from_features(features))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": MODEL_TYPE,
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LearnedRadarAssociationModel":
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"unsupported learned radar association model type {payload.get('model_type')!r}")
        return cls(
            feature_names=tuple(payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            intercept=float(payload["intercept"]),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "LearnedRadarAssociationModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(_jsonable(self.to_dict()), indent=2, allow_nan=False),
            encoding="utf-8",
        )


def radar_association_feature_frame(
    candidates: pd.DataFrame,
    *,
    tracker_state: np.ndarray,
    current_track_id: int | None,
) -> pd.DataFrame:
    """Build learned-association features for one radar frame."""

    if candidates.empty:
        return pd.DataFrame(columns=list(DEFAULT_FEATURE_NAMES))
    state = np.asarray(tracker_state, dtype=float).reshape(6)
    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = positions - state[:3].reshape(1, 3)
    horizontal_residual = np.linalg.norm(residuals[:, :2], axis=1)
    position_residual = np.linalg.norm(residuals, axis=1)

    nis = _numeric_column(candidates, "association_nis")
    finite_nis = np.where(np.isfinite(nis), nis, np.inf)
    ranks = pd.Series(finite_nis).rank(method="average").to_numpy(dtype=float) - 1.0
    ranks = np.where(np.isfinite(finite_nis), ranks, float(len(candidates)))

    cat_prob = _numeric_column(candidates, "cat_prob_uav")
    missing_cat_prob = ~np.isfinite(cat_prob)
    clipped_cat_prob = np.clip(np.where(missing_cat_prob, 0.5, cat_prob), 1.0e-6, 1.0)

    velocity, missing_velocity = _velocity_matrix_enu(candidates)
    velocity_residual = np.linalg.norm(velocity - state[3:6].reshape(1, 3), axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    velocity_residual = np.where(missing_velocity, np.nan, velocity_residual)
    speed = np.where(missing_velocity, np.nan, speed)

    track_id = _numeric_column(candidates, "track_id")
    missing_track_id = ~np.isfinite(track_id)
    if current_track_id is None:
        same_track_id = np.zeros(len(candidates), dtype=float)
        track_id_changed = np.zeros(len(candidates), dtype=float)
    else:
        finite = np.isfinite(track_id)
        same = finite & (track_id.astype(float) == float(current_track_id))
        same_track_id = same.astype(float)
        track_id_changed = (~same & finite).astype(float)

    range_m = _numeric_column(candidates, "range_m")
    radial_velocity = _numeric_column(candidates, "radial_velocity_mps")
    num_inliers = _numeric_column(candidates, "num_inliers")
    return pd.DataFrame(
        {
            "log1p_association_nis": np.log1p(np.where(np.isfinite(nis), np.maximum(nis, 0.0), np.nan)),
            "sqrt_association_nis": np.sqrt(np.where(np.isfinite(nis), np.maximum(nis, 0.0), np.nan)),
            "position_residual_norm_m": position_residual,
            "horizontal_residual_norm_m": horizontal_residual,
            "abs_vertical_residual_m": np.abs(residuals[:, 2]),
            "candidate_count": np.full(len(candidates), float(len(candidates))),
            "candidate_rank_by_nis": ranks,
            "cat_prob_uav": np.where(missing_cat_prob, np.nan, clipped_cat_prob),
            "neg_log_cat_prob_uav": -np.log(clipped_cat_prob),
            "missing_cat_prob_uav": missing_cat_prob.astype(float),
            "track_id_changed": track_id_changed,
            "same_track_id": same_track_id,
            "missing_track_id": missing_track_id.astype(float),
            "velocity_residual_norm_mps": velocity_residual,
            "speed_mps": speed,
            "missing_velocity": missing_velocity.astype(float),
            "log1p_range_m": np.log1p(np.where(np.isfinite(range_m), np.maximum(range_m, 0.0), np.nan)),
            "abs_radial_velocity_mps": np.abs(radial_velocity),
            "missing_radial_velocity": (~np.isfinite(radial_velocity)).astype(float),
            "log1p_num_inliers": np.log1p(np.where(np.isfinite(num_inliers), np.maximum(num_inliers, 0.0), np.nan)),
            "missing_num_inliers": (~np.isfinite(num_inliers)).astype(float),
        }
    )


def score_radar_candidates_with_learned_likelihood(
    candidates: pd.DataFrame,
    *,
    model: LearnedRadarAssociationModel,
    tracker_state: np.ndarray,
    current_track_id: int | None,
) -> pd.DataFrame:
    """Append learned probability and NLL association score columns."""

    features = radar_association_feature_frame(
        candidates,
        tracker_state=tracker_state,
        current_track_id=current_track_id,
    )
    probabilities = np.clip(model.predict_proba_features(features), 1.0e-12, 1.0)
    scored = candidates.copy()
    scored["association_mode"] = "learned-likelihood"
    scored["association_action"] = "learned_likelihood"
    scored["association_learned_probability"] = probabilities
    scored["association_score"] = -np.log(probabilities)
    if "association_candidate_rows" not in scored.columns:
        scored["association_candidate_rows"] = int(len(scored))
    return scored


def fit_learned_radar_association_model(
    examples: pd.DataFrame,
    *,
    feature_names: Iterable[str] = DEFAULT_FEATURE_NAMES,
    label_column: str = "label",
    l2: float = 1.0e-3,
    max_iter: int = 500,
    balance_classes: bool = True,
    metadata: dict[str, Any] | None = None,
) -> LearnedRadarAssociationModel:
    """Fit a standardized logistic association model."""

    names = tuple(str(name) for name in feature_names)
    labels = pd.to_numeric(examples[label_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(labels)
    labels = (labels[keep] > 0.0).astype(float)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        raise ValueError("training examples must contain both positive and negative labels")
    raw = _raw_feature_matrix(examples.loc[keep], names)
    mean = np.nanmean(raw, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(raw), raw, mean.reshape(1, -1))
    scale = np.std(filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-12), scale, 1.0)
    matrix = (filled - mean.reshape(1, -1)) / scale.reshape(1, -1)

    sample_weight = np.ones(labels.size, dtype=float)
    if balance_classes:
        positives = float(labels.sum())
        negatives = float(labels.size - positives)
        sample_weight = np.where(
            labels > 0.0,
            labels.size / max(2.0 * positives, 1.0),
            labels.size / max(2.0 * negatives, 1.0),
        )
    sample_weight /= max(float(sample_weight.sum()), 1.0)

    prior = np.clip(float(labels.mean()), 1.0e-6, 1.0 - 1.0e-6)
    initial = np.zeros(len(names) + 1, dtype=float)
    initial[-1] = np.log(prior / (1.0 - prior))
    result = minimize(
        _loss_and_grad,
        initial,
        args=(matrix, labels, sample_weight, float(l2)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter)},
    )
    if not result.success:
        raise RuntimeError(f"learned radar association fit failed: {result.message}")
    return LearnedRadarAssociationModel(
        feature_names=names,
        mean=mean,
        scale=scale,
        weights=np.asarray(result.x[:-1], dtype=float),
        intercept=float(result.x[-1]),
        metadata=dict(metadata or {}),
    )


def estimate_stateful_transition_costs(
    examples: pd.DataFrame,
    *,
    label_column: str = "label",
    smoothing: float = 1.0,
    min_cost: float = 0.0,
    max_cost: float = 12.0,
) -> dict[str, Any]:
    """Estimate stateful decoder penalties from truth-labeled association rows.

    The learned per-candidate likelihood already captures most geometric and
    Fortem-track evidence.  The stateful decoder still needs discrete costs for
    missed detections, consecutive misses, track-ID switches, and missing track
    IDs.  This helper turns the training labels into empirical log-odds costs so
    the decoder can use data-derived penalties instead of hand-tuned defaults.
    """

    if examples.empty:
        return _empty_stateful_cost_metadata(smoothing=smoothing)
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")
    if max_cost < min_cost:
        raise ValueError("max_cost must be greater than or equal to min_cost")
    if label_column not in examples.columns:
        raise ValueError(f"missing label column {label_column!r}")

    labels = pd.to_numeric(examples[label_column], errors="coerce").fillna(0.0) > 0.0
    frame_keys = _stateful_frame_keys(examples)
    frame_labels = pd.DataFrame(
        {
            "flight": frame_keys["flight"],
            "frame_key": frame_keys["frame_key"],
            "time_s": frame_keys["time_s"],
            "label": labels.to_numpy(dtype=bool),
        }
    )
    frame_summary = (
        frame_labels.groupby(["flight", "frame_key"], dropna=False)
        .agg(label=("label", "max"), time_s=("time_s", "median"))
        .reset_index()
        .sort_values(["flight", "time_s", "frame_key"], kind="mergesort")
    )
    positive_frames = int(frame_summary["label"].sum())
    missed_frames = int(len(frame_summary) - positive_frames)
    missed_detection_cost = _log_odds_cost(
        positive_frames,
        missed_frames,
        smoothing=smoothing,
        min_cost=min_cost,
        max_cost=max_cost,
    )

    continued_misses = 0
    recoveries_after_miss = 0
    for _, group in frame_summary.groupby("flight", dropna=False, sort=False):
        values = group["label"].to_numpy(dtype=bool)
        if values.size < 2:
            continue
        previous_missed = ~values[:-1]
        continued_misses += int(np.sum(previous_missed & ~values[1:]))
        recoveries_after_miss += int(np.sum(previous_missed & values[1:]))
    consecutive_miss_cost = _log_odds_cost(
        recoveries_after_miss,
        continued_misses,
        smoothing=smoothing,
        min_cost=min_cost,
        max_cost=max_cost,
    )

    positive_rows = examples.loc[labels].copy()
    if positive_rows.empty:
        finite_track_count = 0
        missing_track_count = 0
        stay_count = 0
        switch_count = 0
    else:
        positive_keys = _stateful_frame_keys(positive_rows)
        positive_rows["_stateful_flight"] = positive_keys["flight"].to_numpy()
        positive_rows["_stateful_frame_key"] = positive_keys["frame_key"].to_numpy()
        positive_rows["_stateful_time_s"] = positive_keys["time_s"].to_numpy()
        track_ids = pd.to_numeric(positive_rows.get("track_id"), errors="coerce")
        positive_rows["_stateful_track_id"] = track_ids
        finite_track_count = int(track_ids.notna().sum())
        missing_track_count = int(track_ids.isna().sum())
        positive_rows = positive_rows.sort_values(
            ["_stateful_flight", "_stateful_time_s", "_stateful_frame_key"],
            kind="mergesort",
        )
        stay_count = 0
        switch_count = 0
        for _, group in positive_rows.groupby("_stateful_flight", dropna=False, sort=False):
            track = group["_stateful_track_id"].to_numpy(dtype=float)
            if track.size < 2:
                continue
            prev = track[:-1]
            curr = track[1:]
            comparable = np.isfinite(prev) & np.isfinite(curr)
            stay_count += int(np.sum(comparable & (prev == curr)))
            switch_count += int(np.sum(comparable & (prev != curr)))

    missing_track_id_cost = _log_odds_cost(
        finite_track_count,
        missing_track_count,
        smoothing=smoothing,
        min_cost=min_cost,
        max_cost=max_cost,
    )
    track_switch_cost = _log_odds_cost(
        stay_count,
        switch_count,
        smoothing=smoothing,
        min_cost=min_cost,
        max_cost=max_cost,
    )

    return {
        "estimator": "empirical-log-odds-v1",
        "smoothing": float(smoothing),
        "min_cost": float(min_cost),
        "max_cost": float(max_cost),
        "missed_detection_cost": float(missed_detection_cost),
        "consecutive_miss_cost": float(consecutive_miss_cost),
        "track_switch_cost": float(track_switch_cost),
        "missing_track_id_cost": float(missing_track_id_cost),
        "positive_frames": positive_frames,
        "missed_frames": missed_frames,
        "recoveries_after_miss": int(recoveries_after_miss),
        "continued_misses": int(continued_misses),
        "finite_positive_track_ids": int(finite_track_count),
        "missing_positive_track_ids": int(missing_track_count),
        "same_positive_track_transitions": int(stay_count),
        "switched_positive_track_transitions": int(switch_count),
    }


def _empty_stateful_cost_metadata(*, smoothing: float) -> dict[str, Any]:
    return {
        "estimator": "empirical-log-odds-v1",
        "smoothing": float(smoothing),
        "missed_detection_cost": 0.0,
        "consecutive_miss_cost": 0.0,
        "track_switch_cost": 0.0,
        "missing_track_id_cost": 0.0,
        "positive_frames": 0,
        "missed_frames": 0,
        "recoveries_after_miss": 0,
        "continued_misses": 0,
        "finite_positive_track_ids": 0,
        "missing_positive_track_ids": 0,
        "same_positive_track_transitions": 0,
        "switched_positive_track_transitions": 0,
    }


def _stateful_frame_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if "flight" in frame.columns:
        flight = frame["flight"].fillna("").astype(str)
    else:
        flight = pd.Series([""] * len(frame), index=frame.index, dtype=object)
    if "frame_index" in frame.columns:
        frame_index = pd.to_numeric(frame["frame_index"], errors="coerce")
    else:
        frame_index = pd.Series(np.nan, index=frame.index, dtype=float)
    if "time_s" in frame.columns:
        time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    else:
        time_s = pd.Series(np.nan, index=frame.index, dtype=float)
    rounded_time = time_s.round(9)
    frame_key = np.where(
        frame_index.notna(),
        frame_index.astype("Int64").astype(str),
        rounded_time.astype(str),
    )
    return pd.DataFrame(
        {
            "flight": flight.to_numpy(),
            "frame_key": pd.Series(frame_key, index=frame.index).to_numpy(),
            "time_s": time_s.fillna(0.0).to_numpy(dtype=float),
        },
        index=frame.index,
    )


def _log_odds_cost(
    preferred_count: int,
    discouraged_count: int,
    *,
    smoothing: float,
    min_cost: float,
    max_cost: float,
) -> float:
    odds = (float(preferred_count) + float(smoothing)) / (
        float(discouraged_count) + float(smoothing)
    )
    cost = math.log(max(odds, 1.0e-12))
    return float(np.clip(cost, float(min_cost), float(max_cost)))


def _loss_and_grad(params: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    weights = params[:-1]
    intercept = params[-1]
    logits = x @ weights + intercept
    prob = _sigmoid(logits)
    eps = 1.0e-12
    loss = -np.sum(w * (y * np.log(prob + eps) + (1.0 - y) * np.log(1.0 - prob + eps)))
    loss += 0.5 * l2 * float(weights @ weights)
    diff = w * (prob - y)
    grad_w = x.T @ diff + l2 * weights
    grad_b = float(np.sum(diff))
    return float(loss), np.concatenate([grad_w, np.array([grad_b])])


def _raw_feature_matrix(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    columns = []
    for name in names:
        if name in frame.columns:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            columns.append(np.full(len(frame), np.nan, dtype=float))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0), dtype=float)


def _numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _velocity_matrix_enu(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = ("velocity_east_mps", "velocity_north_mps", "velocity_down_mps")
    if not all(column in frame.columns for column in required):
        return np.zeros((len(frame), 3), dtype=float), np.ones(len(frame), dtype=bool)
    velocity = np.column_stack(
        [
            _numeric_column(frame, "velocity_east_mps"),
            _numeric_column(frame, "velocity_north_mps"),
            -_numeric_column(frame, "velocity_down_mps"),
        ]
    )
    missing = ~np.isfinite(velocity).all(axis=1)
    return np.where(np.isfinite(velocity), velocity, 0.0), missing


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    probabilities = np.empty_like(value, dtype=float)
    nonnegative = value >= 0.0

    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    probabilities[~nonnegative] = exp_value / (1.0 + exp_value)
    return probabilities


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
