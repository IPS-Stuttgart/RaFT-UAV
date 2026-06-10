"""Tracklet-level features and dependency-light classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PositionColumns = ("east_m", "north_m", "up_m")


@dataclass(frozen=True)
class StandardizedLogisticModel:
    """Small standardized logistic model for tracklets or calibrated priors."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = _feature_matrix(frame, self.feature_names)
        filled = np.where(np.isfinite(x), x, self.mean.reshape(1, -1))
        z = (filled - self.mean.reshape(1, -1)) / self.scale.reshape(1, -1)
        return _sigmoid(z @ self.weights + self.intercept)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "raft-uav.standardized-logistic-v1",
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "intercept": float(self.intercept),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StandardizedLogisticModel":
        return cls(
            feature_names=tuple(str(x) for x in payload["feature_names"]),
            mean=np.asarray(payload["mean"], dtype=float),
            scale=np.asarray(payload["scale"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            intercept=float(payload["intercept"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StandardizedLogisticModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def tracklet_feature_frame(
    radar: pd.DataFrame,
    *,
    max_frame_gap: float = 1.5,
) -> pd.DataFrame:
    """Aggregate row-level radar detections into Fortem-tracklet features."""

    if radar.empty or "track_id" not in radar.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_key = "frame_index" if "frame_index" in radar.columns else "time_s"
    for track_id, track_rows in radar.groupby("track_id", sort=True):
        ordered = track_rows.sort_values([group_key, "time_s"]).reset_index(drop=True)
        values = pd.to_numeric(ordered[group_key], errors="coerce").to_numpy(dtype=float)
        split_indices = np.r_[0, np.where(np.diff(values) > float(max_frame_gap))[0] + 1, len(ordered)]
        for segment_index, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if segment.empty:
                continue
            rows.append(_tracklet_features(segment, int(track_id), int(segment_index)))
    return pd.DataFrame.from_records(rows)


def frame_context_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Append multi-object context features to each candidate in one radar frame."""

    if candidates.empty:
        return candidates.copy()
    out = candidates.copy()
    positions = out.loc[:, PositionColumns].to_numpy(dtype=float)
    candidate_count = len(out)
    out["frame_candidate_count"] = int(candidate_count)
    if candidate_count > 1:
        distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        distances[distances == 0.0] = np.nan
        out["nearest_neighbor_distance_m"] = np.nanmin(distances, axis=1)
        out["mean_neighbor_distance_m"] = np.nanmean(distances, axis=1)
    else:
        out["nearest_neighbor_distance_m"] = np.nan
        out["mean_neighbor_distance_m"] = np.nan
    if "cat_prob_uav" in out.columns:
        probs = pd.to_numeric(out["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
        out["frame_catprob_rank"] = pd.Series(-probs).rank(method="average").to_numpy(dtype=float) - 1.0
        out["frame_catprob_margin_to_best"] = float(np.nanmax(probs)) - probs
    return out


def fit_logistic_model(
    examples: pd.DataFrame,
    labels: Sequence[int | float] | str,
    *,
    feature_names: Iterable[str] | None = None,
    l2: float = 1.0e-3,
) -> StandardizedLogisticModel:
    """Fit a small standardized logistic model with L2 regularization."""

    if isinstance(labels, str):
        y = pd.to_numeric(examples[labels], errors="coerce").to_numpy(dtype=float)
    else:
        y = np.asarray(labels, dtype=float).reshape(-1)
    if feature_names is None:
        feature_names = [
            col
            for col in examples.columns
            if col != labels and pd.api.types.is_numeric_dtype(examples[col])
        ]
    names = tuple(str(name) for name in feature_names)
    keep = np.isfinite(y)
    y = (y[keep] > 0.0).astype(float)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        raise ValueError("training data must contain both classes")
    raw = _feature_matrix(examples.loc[keep], names)
    mean = np.nanmean(raw, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = np.where(np.isfinite(raw), raw, mean.reshape(1, -1))
    scale = np.std(filled, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    x = (filled - mean.reshape(1, -1)) / scale.reshape(1, -1)
    prior = np.clip(float(y.mean()), 1e-6, 1.0 - 1e-6)
    initial = np.zeros(len(names) + 1)
    initial[-1] = np.log(prior / (1.0 - prior))

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = params[:-1]
        b = params[-1]
        p = _sigmoid(x @ w + b)
        loss = -np.mean(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12))
        loss += 0.5 * float(l2) * float(w @ w)
        diff = (p - y) / max(len(y), 1)
        grad = np.r_[x.T @ diff + float(l2) * w, np.sum(diff)]
        return float(loss), grad

    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return StandardizedLogisticModel(names, mean, scale, result.x[:-1], float(result.x[-1]))


def fit_platt_scaler(probabilities: Sequence[float], labels: Sequence[int | float]) -> StandardizedLogisticModel:
    """Calibrate raw class probabilities with a one-feature Platt scaler."""

    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
    examples = pd.DataFrame({"logit_probability": np.log(p / (1.0 - p)), "label": labels})
    return fit_logistic_model(examples, "label", feature_names=("logit_probability",), l2=1e-6)


def estimate_frame_clutter_density(radar: pd.DataFrame) -> dict[str, float]:
    """Estimate simple frame-level clutter statistics for PDA/MHT priors."""

    if radar.empty:
        return {"mean_candidates_per_frame": 0.0, "p95_candidates_per_frame": 0.0}
    group_key = "frame_index" if "frame_index" in radar.columns else "time_s"
    counts = radar.groupby(group_key).size().to_numpy(dtype=float)
    out = {
        "mean_candidates_per_frame": float(np.mean(counts)),
        "p95_candidates_per_frame": float(np.percentile(counts, 95)),
    }
    if "cat_prob_uav" in radar.columns:
        probs = pd.to_numeric(radar["cat_prob_uav"], errors="coerce").dropna().to_numpy(dtype=float)
        if probs.size:
            out["mean_cat_prob_uav"] = float(np.mean(probs))
            out["low_cat_prob_rate"] = float(np.mean(probs < 0.4))
    return out


def _tracklet_features(segment: pd.DataFrame, track_id: int, segment_index: int) -> dict[str, object]:
    times = pd.to_numeric(segment["time_s"], errors="coerce").to_numpy(dtype=float)
    positions = segment.loc[:, PositionColumns].to_numpy(dtype=float)
    dt = np.diff(times)
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1) if len(segment) > 1 else np.empty(0)
    speeds = np.divide(displacement, dt, out=np.zeros_like(displacement), where=dt > 1e-9)
    catprob = (
        pd.to_numeric(segment["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
        if "cat_prob_uav" in segment.columns
        else np.ones(len(segment), dtype=float)
    )
    ranges = np.linalg.norm(positions, axis=1)
    return {
        "track_id": track_id,
        "segment_index": segment_index,
        "start_time_s": float(np.nanmin(times)),
        "end_time_s": float(np.nanmax(times)),
        "duration_s": float(np.nanmax(times) - np.nanmin(times)) if times.size else 0.0,
        "frames": int(len(segment)),
        "mean_cat_prob_uav": float(np.nanmean(catprob)),
        "min_cat_prob_uav": float(np.nanmin(catprob)),
        "std_cat_prob_uav": float(np.nanstd(catprob)),
        "mean_speed_mps": float(np.nanmean(speeds)) if speeds.size else 0.0,
        "max_speed_mps": float(np.nanmax(speeds)) if speeds.size else 0.0,
        "mean_range_m": float(np.nanmean(ranges)),
        "range_span_m": float(np.nanmax(ranges) - np.nanmin(ranges)),
        "start_east_m": float(positions[0, 0]),
        "start_north_m": float(positions[0, 1]),
        "end_east_m": float(positions[-1, 0]),
        "end_north_m": float(positions[-1, 1]),
    }


def _feature_matrix(frame: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    columns = []
    for name in names:
        if name in frame.columns:
            columns.append(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))
        else:
            columns.append(np.full(len(frame), np.nan))
    return np.column_stack(columns) if columns else np.empty((len(frame), 0))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    probabilities = np.empty_like(value, dtype=float)
    nonnegative = value >= 0.0

    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exp_value = np.exp(value[~nonnegative])
    probabilities[~nonnegative] = exp_value / (1.0 + exp_value)
    return probabilities
