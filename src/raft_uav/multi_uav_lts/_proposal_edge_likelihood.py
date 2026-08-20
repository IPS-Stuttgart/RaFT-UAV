"""Calibrated edge likelihoods and swarm-relative features for LTS tracking."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree

from ._records import Detection, box_iou

_MODEL_SCHEMA = "raft-uav-multi-uav-lts-edge-likelihood-v1"
_SCALE_EPS = 1e-9
EDGE_FEATURE_NAMES = (
    "center_residual",
    "size_log_change",
    "iou_loss",
    "confidence_deficit",
    "gap_frames",
    "swarm_relative_error",
    "swarm_support",
    "density_log_change",
)
SWARM_RELATIVE_FEATURE_INDEX = EDGE_FEATURE_NAMES.index("swarm_relative_error")


@dataclass(frozen=True)
class EdgeFeatureContext:
    """Precomputed local-constellation descriptors for proposal rows."""

    descriptors: Mapping[Detection, tuple[tuple[float, float], ...]]
    neighbor_count: int
    unmatched_penalty: float

    def swarm_features(self, left: Detection, right: Detection) -> tuple[float, float, float]:
        left_descriptor = self.descriptors.get(left, ())
        right_descriptor = self.descriptors.get(right, ())
        left_count = len(left_descriptor)
        right_count = len(right_descriptor)
        density_change = abs(math.log1p(left_count) - math.log1p(right_count))
        if left_count == 0 and right_count == 0:
            return 0.0, 0.0, density_change
        if left_count == 0 or right_count == 0:
            return self.unmatched_penalty, 0.0, density_change

        left_array = np.asarray(left_descriptor, dtype=float)
        right_array = np.asarray(right_descriptor, dtype=float)
        pair_cost = np.linalg.norm(
            left_array[:, None, :] - right_array[None, :, :],
            axis=2,
        )
        pair_cost = np.minimum(pair_cost, self.unmatched_penalty)
        rows, columns = linear_sum_assignment(pair_cost)
        matched_cost = float(pair_cost[rows, columns].sum())
        unmatched_count = abs(left_count - right_count)
        normalized_cost = (
            matched_cost + unmatched_count * self.unmatched_penalty
        ) / max(left_count, right_count)
        support = min(left_count, right_count) / self.neighbor_count
        return normalized_cost, support, density_change


@dataclass(frozen=True)
class EdgeLikelihoodModel:
    """Standardized logistic same-identity edge model."""

    schema: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_example_count: int
    positive_example_count: int
    negative_example_count: int
    sequence_count: int
    l2_penalty: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.schema != _MODEL_SCHEMA:
            raise ValueError(f"unsupported edge-likelihood schema: {self.schema}")
        if self.feature_names != EDGE_FEATURE_NAMES:
            raise ValueError("edge-likelihood feature schema does not match this runtime")
        width = len(EDGE_FEATURE_NAMES)
        for name, values in (
            ("means", self.means),
            ("scales", self.scales),
            ("coefficients", self.coefficients),
        ):
            if len(values) != width:
                raise ValueError(f"edge-likelihood {name} has the wrong length")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"edge-likelihood {name} must be finite")
        if not all(float(value) > 0.0 for value in self.scales):
            raise ValueError("edge-likelihood scales must be positive")
        if not math.isfinite(float(self.intercept)):
            raise ValueError("edge-likelihood intercept must be finite")
        if not math.isfinite(float(self.l2_penalty)) or self.l2_penalty < 0.0:
            raise ValueError("edge-likelihood l2_penalty must be finite and non-negative")
        if self.training_example_count <= 0:
            raise ValueError("edge-likelihood training_example_count must be positive")
        if self.positive_example_count <= 0 or self.negative_example_count <= 0:
            raise ValueError("edge-likelihood training data must contain both classes")
        if (
            self.positive_example_count + self.negative_example_count
            != self.training_example_count
        ):
            raise ValueError("edge-likelihood class counts do not match the total")
        if self.sequence_count <= 0:
            raise ValueError("edge-likelihood sequence_count must be positive")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("edge-likelihood metadata must be an object")

    def logit(self, features: Sequence[float]) -> float:
        raw = np.asarray(features, dtype=float)
        if raw.shape != (len(EDGE_FEATURE_NAMES),):
            raise ValueError("edge feature vector has the wrong shape")
        if not np.isfinite(raw).all():
            raise ValueError("edge feature vector must be finite")
        standardized = (raw - np.asarray(self.means)) / np.asarray(self.scales)
        return float(self.intercept + standardized @ np.asarray(self.coefficients))

    def probability(self, features: Sequence[float]) -> float:
        value = self.logit(features)
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    def negative_log_probability(self, features: Sequence[float]) -> float:
        return float(np.logaddexp(0.0, -self.logit(features)))

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema": self.schema,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "training_example_count": self.training_example_count,
            "positive_example_count": self.positive_example_count,
            "negative_example_count": self.negative_example_count,
            "sequence_count": self.sequence_count,
            "l2_penalty": self.l2_penalty,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EdgeLikelihoodModel:
        try:
            model = cls(
                schema=str(payload["schema"]),
                feature_names=tuple(str(value) for value in payload["feature_names"]),
                means=tuple(float(value) for value in payload["means"]),
                scales=tuple(float(value) for value in payload["scales"]),
                coefficients=tuple(float(value) for value in payload["coefficients"]),
                intercept=float(payload["intercept"]),
                training_example_count=_strict_positive_int(
                    payload["training_example_count"],
                    name="training_example_count",
                ),
                positive_example_count=_strict_positive_int(
                    payload["positive_example_count"],
                    name="positive_example_count",
                ),
                negative_example_count=_strict_positive_int(
                    payload["negative_example_count"],
                    name="negative_example_count",
                ),
                sequence_count=_strict_positive_int(
                    payload["sequence_count"],
                    name="sequence_count",
                ),
                l2_penalty=float(payload["l2_penalty"]),
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed edge-likelihood model") from exc
        model.validate()
        return model


def build_edge_feature_context(
    rows: Sequence[Detection],
    *,
    neighbor_count: int = 4,
    radius_scale: float = 12.0,
    unmatched_penalty: float = 2.0,
) -> EdgeFeatureContext:
    """Build permutation-tolerant local swarm descriptors for every row."""

    neighbors = _positive_int(neighbor_count, name="neighbor_count")
    radius = _positive_finite(radius_scale, name="radius_scale")
    unmatched = _positive_finite(unmatched_penalty, name="unmatched_penalty")
    frames: dict[int, list[Detection]] = {}
    for row in rows:
        frames.setdefault(row.frame_id, []).append(row)

    descriptors: dict[Detection, tuple[tuple[float, float], ...]] = {}
    for frame_rows in frames.values():
        centers = np.asarray(
            [(row.center_x, row.center_y) for row in frame_rows],
            dtype=float,
        )
        tree = cKDTree(centers) if len(frame_rows) > 1 else None
        for row_index, row in enumerate(frame_rows):
            scale = _box_scale(row, row)
            nearby = (
                tree.query_ball_point(centers[row_index], radius * scale)
                if tree is not None
                else ()
            )
            candidates: list[tuple[float, float, float, int]] = []
            for other_index in nearby:
                if other_index == row_index:
                    continue
                other = frame_rows[int(other_index)]
                dx = (other.center_x - row.center_x) / scale
                dy = (other.center_y - row.center_y) / scale
                distance = math.hypot(dx, dy)
                candidates.append((distance, dx, dy, other.object_id))
            candidates.sort(key=lambda value: (value[0], value[1], value[2], value[3]))
            descriptors[row] = tuple(
                (dx, dy) for _distance, dx, dy, _object_id in candidates[:neighbors]
            )
    return EdgeFeatureContext(descriptors, neighbors, unmatched)


def edge_feature_vector(
    left: Detection,
    right: Detection,
    predicted: Detection,
    *,
    gap_frames: int,
    context: EdgeFeatureContext,
) -> np.ndarray:
    """Compute the fixed runtime feature vector for one candidate edge."""

    if isinstance(gap_frames, bool) or not isinstance(gap_frames, int):
        raise ValueError("gap_frames must be an integer")
    if gap_frames < 0:
        raise ValueError("gap_frames must be non-negative")
    scale = _box_scale(predicted, right)
    center = math.hypot(
        right.center_x - predicted.center_x,
        right.center_y - predicted.center_y,
    ) / scale
    size = abs(math.log(right.width / predicted.width)) + abs(
        math.log(right.height / predicted.height)
    )
    swarm_error, swarm_support, density_change = context.swarm_features(left, right)
    features = np.asarray(
        (
            center,
            size,
            1.0 - box_iou(predicted, right),
            max(0.0, 1.0 - right.confidence),
            float(gap_frames),
            swarm_error,
            swarm_support,
            density_change,
        ),
        dtype=float,
    )
    if not np.isfinite(features).all():
        raise ValueError("computed edge features must be finite")
    return features


def fit_edge_likelihood(
    features: Sequence[Sequence[float]],
    labels: Sequence[int | bool],
    *,
    sequence_ids: Sequence[str] | None = None,
    l2_penalty: float = 1.0,
    max_iterations: int = 500,
    metadata: Mapping[str, object] | None = None,
) -> EdgeLikelihoodModel:
    """Fit a sequence-balanced, class-balanced logistic edge model."""

    penalty = _nonnegative_finite(l2_penalty, name="l2_penalty")
    iterations = _positive_int(max_iterations, name="max_iterations")
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(EDGE_FEATURE_NAMES):
        raise ValueError("training feature matrix has the wrong shape")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("training feature matrix must be non-empty and finite")
    target = np.asarray(labels, dtype=int)
    if target.shape != (matrix.shape[0],) or not np.isin(target, (0, 1)).all():
        raise ValueError("training labels must be a binary vector")
    positive_count = int(target.sum())
    negative_count = int(target.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("training labels must contain both classes")

    if sequence_ids is None:
        sequences = tuple("all" for _ in range(matrix.shape[0]))
    else:
        sequences = tuple(str(value) for value in sequence_ids)
        if len(sequences) != matrix.shape[0] or any(not value for value in sequences):
            raise ValueError("sequence_ids must provide one non-empty value per example")
    sequence_counts: dict[str, int] = {}
    for sequence in sequences:
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
    base_weight = np.asarray(
        [1.0 / sequence_counts[sequence] for sequence in sequences],
        dtype=float,
    )
    positive_weight = float(base_weight[target == 1].sum())
    negative_weight = float(base_weight[target == 0].sum())
    sample_weight = base_weight.copy()
    sample_weight[target == 1] *= 0.5 / positive_weight
    sample_weight[target == 0] *= 0.5 / negative_weight
    sample_weight *= matrix.shape[0] / sample_weight.sum()

    means = np.average(matrix, axis=0, weights=base_weight)
    centered = matrix - means
    variances = np.average(centered * centered, axis=0, weights=base_weight)
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales = np.where(scales < 1e-6, 1.0, scales)
    standardized = centered / scales

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = intercept + standardized @ coefficients
        losses = np.logaddexp(0.0, logits) - target * logits
        value = float(
            np.dot(sample_weight, losses) / sample_weight.sum()
            + 0.5 * penalty * np.dot(coefficients, coefficients)
        )
        residual = sample_weight * (_sigmoid(logits) - target) / sample_weight.sum()
        gradient = np.empty_like(parameters)
        gradient[0] = residual.sum()
        gradient[1:] = standardized.T @ residual + penalty * coefficients
        return value, gradient

    initial = np.zeros(len(EDGE_FEATURE_NAMES) + 1, dtype=float)
    result = minimize(
        lambda value: objective(value)[0],
        initial,
        jac=lambda value: objective(value)[1],
        method="L-BFGS-B",
        options={"maxiter": iterations},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"edge-likelihood optimization failed: {result.message}")

    fit_metadata = dict(metadata or {})
    fit_metadata["optimizer"] = {
        "method": "L-BFGS-B",
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
    }
    model = EdgeLikelihoodModel(
        schema=_MODEL_SCHEMA,
        feature_names=EDGE_FEATURE_NAMES,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in result.x[1:]),
        intercept=float(result.x[0]),
        training_example_count=int(target.size),
        positive_example_count=positive_count,
        negative_example_count=negative_count,
        sequence_count=len(sequence_counts),
        l2_penalty=penalty,
        metadata=fit_metadata,
    )
    model.validate()
    return model


def load_edge_likelihood_model(path: Path) -> EdgeLikelihoodModel:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read edge-likelihood model: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("edge-likelihood model must contain a JSON object")
    return EdgeLikelihoodModel.from_dict(payload)


def write_edge_likelihood_model(model: EdgeLikelihoodModel, path: Path) -> None:
    model.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponential = np.exp(values[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _box_scale(left: Detection, right: Detection) -> float:
    return max(
        4.0,
        math.sqrt(max(_SCALE_EPS, left.width * left.height)),
        math.sqrt(max(_SCALE_EPS, right.width * right.height)),
    )


def _strict_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    return _strict_positive_int(value, name=name)


def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return parsed
