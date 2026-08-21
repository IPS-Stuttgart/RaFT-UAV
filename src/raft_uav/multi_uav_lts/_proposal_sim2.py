"""Robust, reliability-gated similarity motion for Multi-UAV LTS proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ._records import Detection

_EPS = 1e-12


@dataclass(frozen=True)
class SimilarityMotionConfig:
    min_pairs: int = 4
    max_rows: int = 96
    max_normalized_step: float = 8.0
    max_normalized_residual: float = 1.5
    max_scale_deviation: float = 0.15
    max_rotation_degrees: float = 15.0
    min_spread_normalized: float = 2.0
    min_residual_improvement: float = 0.05

    def validate(self) -> None:
        _positive_int(self.min_pairs, name="min_pairs")
        _positive_int(self.max_rows, name="max_rows")
        if self.max_rows < self.min_pairs:
            raise ValueError("max_rows must be at least min_pairs")
        _positive(self.max_normalized_step, name="max_normalized_step")
        _positive(self.max_normalized_residual, name="max_normalized_residual")
        _nonnegative(self.max_scale_deviation, name="max_scale_deviation")
        _nonnegative(self.max_rotation_degrees, name="max_rotation_degrees")
        _nonnegative(self.min_spread_normalized, name="min_spread_normalized")
        _nonnegative(
            self.min_residual_improvement,
            name="min_residual_improvement",
        )


@dataclass(frozen=True)
class SimilarityTransform:
    """Transform ``p' = scale * R * p + translation``."""

    scale: float = 1.0
    cosine: float = 1.0
    sine: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0

    @classmethod
    def identity(cls) -> "SimilarityTransform":
        return cls()

    @property
    def rotation_degrees(self) -> float:
        return math.degrees(math.atan2(self.sine, self.cosine))

    def apply_xy(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.scale * (self.cosine * x - self.sine * y) + self.translation_x,
            self.scale * (self.sine * x + self.cosine * y) + self.translation_y,
        )

    def apply_detection(self, row: Detection) -> Detection:
        center_x, center_y = self.apply_xy(row.center_x, row.center_y)
        width = row.width * self.scale
        height = row.height * self.scale
        return replace(
            row,
            x1=center_x - 0.5 * width,
            y1=center_y - 0.5 * height,
            width=width,
            height=height,
        )

    def inverse(self) -> "SimilarityTransform":
        if not math.isfinite(self.scale) or self.scale <= _EPS:
            raise ValueError("similarity scale must be finite and positive")
        inverse_scale = 1.0 / self.scale
        return SimilarityTransform(
            scale=inverse_scale,
            cosine=self.cosine,
            sine=-self.sine,
            translation_x=-inverse_scale
            * (self.cosine * self.translation_x + self.sine * self.translation_y),
            translation_y=inverse_scale
            * (self.sine * self.translation_x - self.cosine * self.translation_y),
        )

    def compose(self, previous: "SimilarityTransform") -> "SimilarityTransform":
        """Return ``self(previous(point))``."""
        cosine = self.cosine * previous.cosine - self.sine * previous.sine
        sine = self.sine * previous.cosine + self.cosine * previous.sine
        translation_x, translation_y = self.apply_xy(
            previous.translation_x,
            previous.translation_y,
        )
        norm = math.hypot(cosine, sine)
        if norm <= _EPS:
            raise ValueError("similarity rotation became degenerate")
        return SimilarityTransform(
            scale=self.scale * previous.scale,
            cosine=cosine / norm,
            sine=sine / norm,
            translation_x=translation_x,
            translation_y=translation_y,
        )


@dataclass(frozen=True)
class SimilarityMotionStep:
    frame_id: int
    transform: SimilarityTransform
    model: str
    pair_count: int
    inlier_count: int
    residual_normalized: float | None
    translation_residual_normalized: float | None
    spread_normalized: float
    fallback_reason: str | None


def estimate_similarity_steps(
    rows: Sequence[Detection],
    config: SimilarityMotionConfig,
) -> dict[int, SimilarityMotionStep]:
    config.validate()
    frames: dict[int, list[Detection]] = {}
    for row in rows:
        frames.setdefault(row.frame_id, []).append(row)
    return {
        frame_id: _estimate_step(
            frame_id,
            frames[frame_id],
            frames[frame_id + 1],
            config,
        )
        for frame_id in sorted(frames)
        if frame_id + 1 in frames
    }


def cumulative_transforms(
    steps: dict[int, SimilarityMotionStep],
    max_frame: int,
) -> dict[int, SimilarityTransform]:
    if isinstance(max_frame, bool) or not isinstance(max_frame, int) or max_frame < 1:
        raise ValueError("max_frame must be a positive integer")
    current = SimilarityTransform.identity()
    result = {1: current}
    for frame_id in range(1, max_frame):
        step = steps.get(frame_id)
        if step is not None:
            current = step.transform.compose(current)
        result[frame_id + 1] = current
    return result


def stabilize_rows(
    rows: Sequence[Detection],
    transforms: dict[int, SimilarityTransform],
) -> tuple[Detection, ...]:
    identity = SimilarityTransform.identity()
    return tuple(
        transforms.get(row.frame_id, identity).inverse().apply_detection(row)
        for row in rows
    )


def restore_rows(
    rows: Sequence[Detection],
    transforms: dict[int, SimilarityTransform],
) -> tuple[Detection, ...]:
    identity = SimilarityTransform.identity()
    return tuple(transforms.get(row.frame_id, identity).apply_detection(row) for row in rows)


def step_summary(steps: dict[int, SimilarityMotionStep]) -> dict[str, int | float | None]:
    values = tuple(steps.values())
    valid = tuple(step for step in values if step.model != "identity")
    residuals = [step.residual_normalized for step in valid if step.residual_normalized is not None]
    return {
        "step_count": len(values),
        "sim2_step_count": sum(step.model == "sim2" for step in values),
        "translation_step_count": sum(step.model == "translation" for step in values),
        "identity_step_count": sum(step.model == "identity" for step in values),
        "median_scale": _median([step.transform.scale for step in valid], 1.0),
        "median_absolute_rotation_degrees": _median(
            [abs(step.transform.rotation_degrees) for step in valid],
            0.0,
        ),
        "median_residual_normalized": _median(residuals, None),
    }


def _estimate_step(
    frame_id: int,
    left_rows: Sequence[Detection],
    right_rows: Sequence[Detection],
    config: SimilarityMotionConfig,
) -> SimilarityMotionStep:
    left = _motion_rows(left_rows, config.max_rows)
    right = _motion_rows(right_rows, config.max_rows)
    if min(len(left), len(right)) < config.min_pairs:
        return _identity_step(frame_id, 0, "insufficient_rows")
    left_points, left_width, left_height = _arrays(left)
    right_points, right_width, right_height = _arrays(right)
    initial = np.median(right_points, axis=0) - np.median(left_points, axis=0)
    delta = right_points[None, :, :] - (
        left_points[:, None, :] + initial[None, None, :]
    )
    scales = np.maximum(
        4.0,
        np.maximum(
            np.sqrt(np.maximum(_EPS, left_width * left_height))[:, None],
            np.sqrt(np.maximum(_EPS, right_width * right_height))[None, :],
        ),
    )
    center_cost = np.linalg.norm(delta, axis=2) / scales
    size_cost = np.abs(np.log(right_width[None, :] / left_width[:, None]))
    size_cost += np.abs(np.log(right_height[None, :] / left_height[:, None]))
    costs = center_cost + 0.25 * size_cost
    forward = np.argmin(costs, axis=1)
    backward = np.argmin(costs, axis=0)
    left_indices = np.arange(len(left), dtype=int)
    reciprocal = backward[forward] == left_indices
    selected_left = left_indices[reciprocal]
    selected_right = forward[reciprocal]
    if len(selected_left) < config.min_pairs:
        return _identity_step(
            frame_id,
            int(len(selected_left)),
            "insufficient_reciprocal_pairs",
        )
    source = left_points[selected_left]
    target = right_points[selected_right]
    pair_scales = scales[selected_left, selected_right]
    plausible = np.linalg.norm(target - source, axis=1) / pair_scales
    plausible = plausible <= config.max_normalized_step
    source = source[plausible]
    target = target[plausible]
    pair_scales = pair_scales[plausible]
    pair_count = int(len(source))
    if pair_count < config.min_pairs:
        return _identity_step(frame_id, pair_count, "implausible_pair_steps")

    translation = np.median(target - source, axis=0)
    translation_residual = np.linalg.norm(
        target - source - translation[None, :], axis=1
    ) / pair_scales
    translation_mask = translation_residual <= config.max_normalized_residual
    translation_valid = int(np.count_nonzero(translation_mask)) >= config.min_pairs
    if translation_valid:
        source_t = source[translation_mask]
        target_t = target[translation_mask]
        scales_t = pair_scales[translation_mask]
        translation = np.median(target_t - source_t, axis=0)
        translation_residual = np.linalg.norm(
            target_t - source_t - translation[None, :], axis=1
        ) / scales_t
        translation_median = float(np.median(translation_residual))
        translation_inliers = int(len(source_t))
    else:
        translation_median = float(np.median(translation_residual))
        translation_inliers = int(np.count_nonzero(translation_mask))
    spread = _spread(source, pair_scales)
    translation_step = SimilarityMotionStep(
        frame_id,
        SimilarityTransform(
            translation_x=float(translation[0]),
            translation_y=float(translation[1]),
        ),
        "translation",
        pair_count,
        translation_inliers,
        translation_median,
        translation_median,
        spread,
        None,
    )

    sim2_reason = "insufficient_spread"
    if spread >= config.min_spread_normalized:
        fitted = _fit_similarity_robust(source, target, pair_scales, config)
        if fitted is None:
            sim2_reason = "sim2_fit_failed"
        else:
            transform, inliers, residual = fitted
            if abs(transform.scale - 1.0) > config.max_scale_deviation:
                sim2_reason = "scale_gate"
            elif abs(transform.rotation_degrees) > config.max_rotation_degrees:
                sim2_reason = "rotation_gate"
            elif translation_median - residual < config.min_residual_improvement:
                sim2_reason = "insufficient_improvement"
            else:
                return SimilarityMotionStep(
                    frame_id,
                    transform,
                    "sim2",
                    pair_count,
                    inliers,
                    residual,
                    translation_median,
                    spread,
                    None,
                )
    if translation_valid:
        return replace(translation_step, fallback_reason=sim2_reason)
    return _identity_step(frame_id, pair_count, "no_reliable_motion_model")


def _fit_similarity_robust(
    source: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    config: SimilarityMotionConfig,
) -> tuple[SimilarityTransform, int, float] | None:
    mask = np.ones(len(source), dtype=bool)
    for _iteration in range(3):
        if int(np.count_nonzero(mask)) < config.min_pairs:
            return None
        transform = _fit_similarity(source[mask], target[mask])
        if transform is None:
            return None
        residual = np.linalg.norm(
            target - _apply_points(transform, source), axis=1
        ) / scales
        next_mask = residual <= config.max_normalized_residual
        if int(np.count_nonzero(next_mask)) < config.min_pairs:
            return None
        if np.array_equal(mask, next_mask):
            break
        mask = next_mask
    transform = _fit_similarity(source[mask], target[mask])
    if transform is None:
        return None
    residual = np.linalg.norm(
        target[mask] - _apply_points(transform, source[mask]), axis=1
    ) / scales[mask]
    median = float(np.median(residual))
    if median > config.max_normalized_residual:
        return None
    return transform, int(np.count_nonzero(mask)), median


def _fit_similarity(
    source: np.ndarray, target: np.ndarray
) -> SimilarityTransform | None:
    if len(source) < 2 or source.shape != target.shape or source.shape[1] != 2:
        return None
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if not math.isfinite(variance) or variance <= _EPS:
        return None
    covariance = target_centered.T @ source_centered / len(source)
    try:
        left, singular, right_transpose = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    correction = np.ones(2, dtype=float)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1] = -1.0
    rotation = left @ np.diag(correction) @ right_transpose
    scale = float(np.sum(singular * correction) / variance)
    if not math.isfinite(scale) or scale <= _EPS:
        return None
    translation = target_mean - scale * (rotation @ source_mean)
    cosine = float(rotation[0, 0])
    sine = float(rotation[1, 0])
    norm = math.hypot(cosine, sine)
    values = (*rotation.ravel(), scale, *translation)
    if norm <= _EPS or not all(math.isfinite(float(value)) for value in values):
        return None
    return SimilarityTransform(
        scale,
        cosine / norm,
        sine / norm,
        float(translation[0]),
        float(translation[1]),
    )


def _apply_points(transform: SimilarityTransform, points: np.ndarray) -> np.ndarray:
    rotation = np.asarray(
        [[transform.cosine, -transform.sine], [transform.sine, transform.cosine]]
    )
    translation = np.asarray([transform.translation_x, transform.translation_y])
    return transform.scale * (points @ rotation.T) + translation[None, :]


def _motion_rows(rows: Sequence[Detection], max_rows: int) -> tuple[Detection, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (-row.confidence, row.x1, row.y1, row.object_id),
        )[:max_rows]
    )


def _arrays(
    rows: Sequence[Detection],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([[row.center_x, row.center_y] for row in rows], dtype=float),
        np.asarray([row.width for row in rows], dtype=float),
        np.asarray([row.height for row in rows], dtype=float),
    )


def _spread(source: np.ndarray, scales: np.ndarray) -> float:
    centered = source - np.mean(source, axis=0)
    rms = math.sqrt(float(np.mean(np.sum(centered**2, axis=1))))
    return rms / max(_EPS, float(np.median(scales)))


def _identity_step(
    frame_id: int, pair_count: int, reason: str
) -> SimilarityMotionStep:
    return SimilarityMotionStep(
        frame_id,
        SimilarityTransform.identity(),
        "identity",
        pair_count,
        0,
        None,
        None,
        0.0,
        reason,
    )


def _median(values: Sequence[float], default: float | None) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return default if not finite else float(np.median(finite))


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real scalar")
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        parsed = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite real scalar")
    return parsed


def _positive(value: object, *, name: str) -> float:
    parsed = _finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative(value: object, *, name: str) -> float:
    parsed = _finite(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return parsed
