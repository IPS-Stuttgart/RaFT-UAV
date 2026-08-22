"""Reliability-gated similarity common motion for Multi-UAV proposal graphs.

The maintained graph tracker models shared image/swarm motion as a translation.
This experimental extension fits an isotropic scale, in-plane rotation, and
translation only when the richer model is well-conditioned and materially
reduces normalized correspondence residuals. Every rejected or degenerate fit
falls back to the exact maintained translation estimate.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Iterator, Mapping, Protocol, Sequence

import numpy as np

from . import _proposal_common_motion as translation_motion
from ._records import Detection, box_iou

_EPS = 1e-12
_MAX_ROWS = 96


class _NodeLike(Protocol):
    index: int
    row: Detection


class _Parameters(Protocol):
    enable_common_motion: bool
    common_motion_min_pairs: int
    common_motion_max_normalized_step: float
    common_motion_max_normalized_residual: float
    center_weight: float
    size_weight: float
    iou_weight: float
    velocity_weight: float
    gap_weight: float
    confidence_weight: float
    max_link_gap: int
    max_link_cost: float


class _TrackletLike(Protocol):
    index: int
    rows: tuple[Detection, ...]
    seed_id: int | None


@dataclass(frozen=True)
class SimilarityMotionConfig:
    """Reliability controls for the optional Sim(2) common-motion fit."""

    min_pairs: int = 4
    max_scale_change: float = 0.12
    max_rotation_deg: float = 10.0
    max_normalized_residual: float = 1.0
    min_normalized_spread: float = 2.0
    min_residual_improvement: float = 0.05
    refinement_iterations: int = 3

    def validate(self) -> None:
        _positive_int(self.min_pairs, name="min_pairs")
        _nonnegative_finite(self.max_scale_change, name="max_scale_change")
        _nonnegative_finite(self.max_rotation_deg, name="max_rotation_deg")
        _positive_finite(
            self.max_normalized_residual,
            name="max_normalized_residual",
        )
        _positive_finite(
            self.min_normalized_spread,
            name="min_normalized_spread",
        )
        _nonnegative_finite(
            self.min_residual_improvement,
            name="min_residual_improvement",
        )
        _positive_int(self.refinement_iterations, name="refinement_iterations")


@dataclass(frozen=True)
class SimilarityTransform:
    """One frame-to-frame isotropic similarity transform."""

    scale: float = 1.0
    cos_theta: float = 1.0
    sin_theta: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    model: str = "identity"
    support: int = 0
    median_normalized_residual: float = 0.0

    @classmethod
    def identity(cls) -> SimilarityTransform:
        return cls()

    @classmethod
    def translation(
        cls,
        dx: float,
        dy: float,
        *,
        support: int = 0,
        median_normalized_residual: float = 0.0,
    ) -> SimilarityTransform:
        return cls(
            tx=float(dx),
            ty=float(dy),
            model="translation",
            support=int(support),
            median_normalized_residual=float(median_normalized_residual),
        )

    @property
    def angle_rad(self) -> float:
        return math.atan2(self.sin_theta, self.cos_theta)

    def __iter__(self) -> Iterator[float]:
        """Preserve tuple-unpacking compatibility with translation-only callers."""
        yield self.tx
        yield self.ty

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        mapped_x, mapped_y = self.apply_vector(x, y)
        return mapped_x + self.tx, mapped_y + self.ty

    def apply_vector(self, x: float, y: float) -> tuple[float, float]:
        return (
            self.scale * (self.cos_theta * x - self.sin_theta * y),
            self.scale * (self.sin_theta * x + self.cos_theta * y),
        )

    def inverse_apply_vector(self, x: float, y: float) -> tuple[float, float]:
        if self.scale <= _EPS:
            raise ValueError("similarity transform scale must be positive")
        inverse_scale = 1.0 / self.scale
        return (
            inverse_scale * (self.cos_theta * x + self.sin_theta * y),
            inverse_scale * (-self.sin_theta * x + self.cos_theta * y),
        )

    def apply_detection(self, row: Detection, frame: int) -> Detection:
        center_x, center_y = self.apply_point(row.center_x, row.center_y)
        width = self.scale * (
            abs(self.cos_theta) * row.width + abs(self.sin_theta) * row.height
        )
        height = self.scale * (
            abs(self.sin_theta) * row.width + abs(self.cos_theta) * row.height
        )
        width = max(_EPS, width)
        height = max(_EPS, height)
        return replace(
            row,
            frame_id=frame,
            x1=center_x - 0.5 * width,
            y1=center_y - 0.5 * height,
            width=width,
            height=height,
        )

    def then(self, following: SimilarityTransform) -> SimilarityTransform:
        """Compose transforms as ``following(self(point))``."""
        tx, ty = following.apply_vector(self.tx, self.ty)
        cos_theta = (
            following.cos_theta * self.cos_theta
            - following.sin_theta * self.sin_theta
        )
        sin_theta = (
            following.sin_theta * self.cos_theta
            + following.cos_theta * self.sin_theta
        )
        norm = math.hypot(cos_theta, sin_theta)
        if norm <= _EPS:
            cos_theta, sin_theta = 1.0, 0.0
        else:
            cos_theta /= norm
            sin_theta /= norm
        model = (
            "similarity"
            if self.model == "similarity" or following.model == "similarity"
            else (
                "translation"
                if self.model == "translation" or following.model == "translation"
                else "identity"
            )
        )
        return SimilarityTransform(
            scale=self.scale * following.scale,
            cos_theta=cos_theta,
            sin_theta=sin_theta,
            tx=tx + following.tx,
            ty=ty + following.ty,
            model=model,
            support=min(self.support, following.support),
            median_normalized_residual=max(
                self.median_normalized_residual,
                following.median_normalized_residual,
            ),
        )


def estimate_common_motion(
    nodes: Sequence[_NodeLike],
    parameters: _Parameters,
    *,
    config: SimilarityMotionConfig,
) -> dict[int, SimilarityTransform]:
    """Fit safe Sim(2) steps, preserving translation as the exact fallback."""
    config.validate()
    fallback = translation_motion.estimate_common_motion(nodes, parameters)
    if not parameters.enable_common_motion:
        return {}

    frames: dict[int, list[_NodeLike]] = {}
    for node in nodes:
        frames.setdefault(node.row.frame_id, []).append(node)

    estimates = {
        frame: SimilarityTransform.translation(dx, dy)
        for frame, (dx, dy) in fallback.items()
    }
    required_pairs = max(config.min_pairs, parameters.common_motion_min_pairs)
    for frame in sorted(frames):
        left = _motion_rows(frames.get(frame, ()))
        right = _motion_rows(frames.get(frame + 1, ()))
        if min(len(left), len(right)) < required_pairs:
            continue
        fallback_step = estimates.get(
            frame,
            _centroid_translation(left, right),
        )
        pairs = _reciprocal_pairs(
            left,
            right,
            fallback_step,
            max_normalized_step=parameters.common_motion_max_normalized_step,
        )
        if len(pairs) < required_pairs:
            continue
        fitted = _robust_similarity(
            pairs,
            fallback_step,
            config=config,
            required_pairs=required_pairs,
        )
        if fitted is not None:
            estimates[frame] = fitted
    return estimates


def transform_between(
    common_motion: Mapping[int, object],
    start_frame: int,
    end_frame: int,
) -> SimilarityTransform:
    if end_frame <= start_frame:
        return SimilarityTransform.identity()
    result = SimilarityTransform.identity()
    for frame in range(start_frame, end_frame):
        step = as_transform(common_motion.get(frame, (0.0, 0.0)))
        result = result.then(step)
    return result


def as_transform(value: object) -> SimilarityTransform:
    if isinstance(value, SimilarityTransform):
        return value
    try:
        dx, dy = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise TypeError("common motion step must be a transform or length-two pair") from exc
    return SimilarityTransform.translation(float(dx), float(dy))


def observation_cost(
    left: Detection,
    right: Detection,
    parameters: _Parameters,
    motion: object = (0.0, 0.0),
) -> float:
    predicted = as_transform(motion).apply_detection(left, right.frame_id)
    scale = _box_scale(predicted, right)
    center = math.hypot(
        right.center_x - predicted.center_x,
        right.center_y - predicted.center_y,
    ) / scale
    size = abs(math.log(right.width / predicted.width)) + abs(
        math.log(right.height / predicted.height)
    )
    return (
        parameters.center_weight * center
        + parameters.size_weight * size
        + parameters.iou_weight * (1.0 - box_iou(predicted, right))
        + parameters.confidence_weight * (1.0 - right.confidence)
    )


def predict(
    rows: tuple[Detection, ...],
    frame: int,
    common_motion: Mapping[int, object],
) -> Detection:
    last = rows[-1]
    if frame <= last.frame_id:
        return replace(last, frame_id=frame)
    interval = transform_between(common_motion, last.frame_id, frame)
    common_prediction = interval.apply_detection(last, frame)
    if len(rows) < 2:
        return common_prediction

    previous = rows[-2]
    history = last.frame_id - previous.frame_id
    if history <= 0:
        return common_prediction
    previous_interval = transform_between(
        common_motion,
        previous.frame_id,
        last.frame_id,
    )
    predicted_previous = previous_interval.apply_detection(previous, last.frame_id)
    residual_vx = (last.center_x - predicted_previous.center_x) / history
    residual_vy = (last.center_y - predicted_previous.center_y) / history
    delta = frame - last.frame_id
    residual_dx, residual_dy = interval.apply_vector(
        residual_vx * delta,
        residual_vy * delta,
    )
    return replace(
        common_prediction,
        x1=common_prediction.center_x
        + residual_dx
        - 0.5 * common_prediction.width,
        y1=common_prediction.center_y
        + residual_dy
        - 0.5 * common_prediction.height,
    )


def velocity(
    rows: tuple[Detection, ...],
    *,
    tail: bool,
    common_motion: Mapping[int, object],
) -> tuple[float, float] | None:
    if len(rows) < 2:
        return None
    left, right = (rows[-2], rows[-1]) if tail else (rows[0], rows[1])
    delta = right.frame_id - left.frame_id
    if delta <= 0:
        return None
    interval = transform_between(common_motion, left.frame_id, right.frame_id)
    predicted_left = interval.apply_detection(left, right.frame_id)
    return (
        (right.center_x - predicted_left.center_x) / delta,
        (right.center_y - predicted_left.center_y) / delta,
    )


def continuation_potential(
    left: _TrackletLike,
    right: _TrackletLike,
    starts: dict[int, list[_TrackletLike]],
    frames: Sequence[int],
    parameters: _Parameters,
    common_motion: Mapping[int, object],
    horizon: int,
) -> float:
    first = left.rows[-1]
    second = right.rows[0]
    delta = second.frame_id - first.frame_id
    if delta <= 0:
        return 0.0
    first_at_second = predict((first,), second.frame_id, common_motion)
    residual_vx = (second.center_x - first_at_second.center_x) / delta
    residual_vy = (second.center_y - first_at_second.center_y) / delta

    lower = bisect_left(frames, second.frame_id + 1)
    upper = bisect_right(frames, second.frame_id + horizon)
    per_frame: list[float] = []
    for frame in frames[lower:upper]:
        future = starts.get(frame, ())
        if not future:
            continue
        best = min(
            future_cost(
                second,
                candidate.rows[0],
                residual_vx,
                residual_vy,
                parameters,
                common_motion,
            )
            for candidate in future
        )
        if math.isfinite(best):
            per_frame.append(best)
    return float(np.mean(per_frame)) if per_frame else 0.0


def future_cost(
    current: Detection,
    future: Detection,
    residual_vx: float,
    residual_vy: float,
    parameters: _Parameters,
    common_motion: Mapping[int, object],
) -> float:
    delta = future.frame_id - current.frame_id
    if delta <= 0:
        return math.inf
    interval = transform_between(common_motion, current.frame_id, future.frame_id)
    common_prediction = interval.apply_detection(current, future.frame_id)
    residual_dx, residual_dy = interval.apply_vector(
        residual_vx * delta,
        residual_vy * delta,
    )
    predicted = replace(
        common_prediction,
        x1=common_prediction.center_x
        + residual_dx
        - 0.5 * common_prediction.width,
        y1=common_prediction.center_y
        + residual_dy
        - 0.5 * common_prediction.height,
    )
    scale = _box_scale(predicted, future)
    center = math.hypot(
        future.center_x - predicted.center_x,
        future.center_y - predicted.center_y,
    ) / scale
    size = abs(math.log(future.width / predicted.width)) + abs(
        math.log(future.height / predicted.height)
    )
    future_residual_x = future.center_x - common_prediction.center_x
    future_residual_y = future.center_y - common_prediction.center_y
    next_vx, next_vy = interval.inverse_apply_vector(
        future_residual_x,
        future_residual_y,
    )
    next_vx /= delta
    next_vy /= delta
    acceleration = math.hypot(
        next_vx - residual_vx,
        next_vy - residual_vy,
    ) / scale
    return (
        parameters.center_weight * center
        + parameters.size_weight * size
        + parameters.iou_weight * (1.0 - box_iou(predicted, future))
        + parameters.velocity_weight * acceleration
        + parameters.confidence_weight * (1.0 - future.confidence)
    )


def path_acceleration(
    path: list[int],
    tracklets: Mapping[int, _TrackletLike],
    common_motion: Mapping[int, object],
    *,
    clip: float,
) -> float:
    seed_ids = {
        tracklets[index].seed_id
        for index in path
        if tracklets[index].seed_id is not None
    }
    if len(seed_ids) > 1:
        return math.inf
    rows = tuple(
        sorted(
            (row for index in path for row in tracklets[index].rows),
            key=lambda row: (row.frame_id, row.center_x, row.center_y),
        )
    )
    if len(rows) < 3:
        return 0.0
    anchor = rows[0]
    residual_positions: list[tuple[float, float]] = []
    for row in rows:
        interval = transform_between(common_motion, anchor.frame_id, row.frame_id)
        common_x, common_y = interval.apply_point(anchor.center_x, anchor.center_y)
        residual_positions.append(
            interval.inverse_apply_vector(
                row.center_x - common_x,
                row.center_y - common_y,
            )
        )

    result = 0.0
    for index in range(2, len(rows)):
        first = rows[index - 2]
        middle = rows[index - 1]
        last = rows[index]
        dt_left = middle.frame_id - first.frame_id
        dt_right = last.frame_id - middle.frame_id
        if dt_left <= 0 or dt_right <= 0:
            continue
        p0 = residual_positions[index - 2]
        p1 = residual_positions[index - 1]
        p2 = residual_positions[index]
        left_velocity = (
            (p1[0] - p0[0]) / dt_left,
            (p1[1] - p0[1]) / dt_left,
        )
        right_velocity = (
            (p2[0] - p1[0]) / dt_right,
            (p2[1] - p1[1]) / dt_right,
        )
        scale = max(4.0, math.sqrt(max(_EPS, middle.width * middle.height)))
        normalized = math.hypot(
            right_velocity[0] - left_velocity[0],
            right_velocity[1] - left_velocity[1],
        ) / scale
        result += min(clip, normalized)
    return result


def _centroid_translation(
    left: Sequence[_NodeLike],
    right: Sequence[_NodeLike],
) -> SimilarityTransform:
    left_x = np.asarray([node.row.center_x for node in left], dtype=float)
    left_y = np.asarray([node.row.center_y for node in left], dtype=float)
    right_x = np.asarray([node.row.center_x for node in right], dtype=float)
    right_y = np.asarray([node.row.center_y for node in right], dtype=float)
    return SimilarityTransform.translation(
        float(np.median(right_x) - np.median(left_x)),
        float(np.median(right_y) - np.median(left_y)),
    )


def _reciprocal_pairs(
    left: Sequence[_NodeLike],
    right: Sequence[_NodeLike],
    fallback: SimilarityTransform,
    *,
    max_normalized_step: float,
) -> tuple[tuple[Detection, Detection, float], ...]:
    left_rows = tuple(node.row for node in left)
    right_rows = tuple(node.row for node in right)
    costs = np.asarray(
        [
            [
                _pair_cost(fallback.apply_detection(a, b.frame_id), b)
                for b in right_rows
            ]
            for a in left_rows
        ],
        dtype=float,
    )
    forward = np.argmin(costs, axis=1)
    backward = np.argmin(costs, axis=0)
    pairs: list[tuple[Detection, Detection, float]] = []
    for left_index, right_index_raw in enumerate(forward):
        right_index = int(right_index_raw)
        if int(backward[right_index]) != left_index:
            continue
        a = left_rows[left_index]
        b = right_rows[right_index]
        pair_scale = _box_scale(a, b)
        raw_step = math.hypot(
            b.center_x - a.center_x,
            b.center_y - a.center_y,
        ) / pair_scale
        if raw_step <= max_normalized_step:
            pairs.append((a, b, pair_scale))
    return tuple(pairs)


def _robust_similarity(
    pairs: tuple[tuple[Detection, Detection, float], ...],
    fallback: SimilarityTransform,
    *,
    config: SimilarityMotionConfig,
    required_pairs: int,
) -> SimilarityTransform | None:
    source = np.asarray(
        [[left.center_x, left.center_y] for left, _right, _scale in pairs],
        dtype=float,
    )
    target = np.asarray(
        [[right.center_x, right.center_y] for _left, right, _scale in pairs],
        dtype=float,
    )
    pair_scale = np.asarray([scale for _left, _right, scale in pairs], dtype=float)
    inliers = np.ones(len(pairs), dtype=bool)
    transform: SimilarityTransform | None = None
    for _ in range(config.refinement_iterations):
        if int(np.count_nonzero(inliers)) < required_pairs:
            return None
        transform = _fit_similarity(source[inliers], target[inliers])
        if transform is None:
            return None
        residual = _normalized_residual(transform, source, target, pair_scale)
        updated = residual <= config.max_normalized_residual
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    if transform is None or int(np.count_nonzero(inliers)) < required_pairs:
        return None
    transform = _fit_similarity(source[inliers], target[inliers])
    if transform is None:
        return None

    residual = _normalized_residual(transform, source, target, pair_scale)
    median_residual = float(np.median(residual[inliers]))
    translation_residual = _normalized_residual(
        fallback,
        source,
        target,
        pair_scale,
    )
    translation_median = float(np.median(translation_residual[inliers]))
    spread = _normalized_spread(source[inliers], pair_scale[inliers])
    rotation_deg = abs(math.degrees(transform.angle_rad))
    if (
        abs(transform.scale - 1.0) > config.max_scale_change
        or rotation_deg > config.max_rotation_deg
        or median_residual > config.max_normalized_residual
        or spread < config.min_normalized_spread
        or translation_median - median_residual
        < config.min_residual_improvement
    ):
        return None
    return replace(
        transform,
        model="similarity",
        support=int(np.count_nonzero(inliers)),
        median_normalized_residual=median_residual,
    )


def _fit_similarity(
    source: np.ndarray,
    target: np.ndarray,
) -> SimilarityTransform | None:
    if len(source) < 2 or source.shape != target.shape or source.shape[1] != 2:
        return None
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.sum(source_centered * source_centered))
    if not math.isfinite(variance) or variance <= _EPS:
        return None
    covariance = source_centered.T @ target_centered
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    correction = np.eye(2)
    if np.linalg.det(vt.T @ u.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= _EPS:
        return None
    translation = target_mean - scale * (rotation @ source_mean)
    cos_theta = float(rotation[0, 0])
    sin_theta = float(rotation[1, 0])
    norm = math.hypot(cos_theta, sin_theta)
    if norm <= _EPS:
        return None
    return SimilarityTransform(
        scale=scale,
        cos_theta=cos_theta / norm,
        sin_theta=sin_theta / norm,
        tx=float(translation[0]),
        ty=float(translation[1]),
        model="similarity",
    )


def _normalized_residual(
    transform: SimilarityTransform,
    source: np.ndarray,
    target: np.ndarray,
    pair_scale: np.ndarray,
) -> np.ndarray:
    linear = np.asarray(
        [
            [transform.cos_theta, -transform.sin_theta],
            [transform.sin_theta, transform.cos_theta],
        ],
        dtype=float,
    )
    predicted = transform.scale * (source @ linear.T)
    predicted[:, 0] += transform.tx
    predicted[:, 1] += transform.ty
    return np.linalg.norm(target - predicted, axis=1) / pair_scale


def _normalized_spread(source: np.ndarray, pair_scale: np.ndarray) -> float:
    centered = source - np.mean(source, axis=0)
    rms = math.sqrt(float(np.mean(np.sum(centered * centered, axis=1))))
    reference = float(np.median(pair_scale))
    return rms / max(_EPS, reference)


def _motion_rows(nodes: Sequence[_NodeLike]) -> tuple[_NodeLike, ...]:
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                -node.row.confidence,
                node.row.x1,
                node.row.y1,
                node.index,
            ),
        )[:_MAX_ROWS]
    )


def _pair_cost(left: Detection, right: Detection) -> float:
    scale = _box_scale(left, right)
    center = math.hypot(
        right.center_x - left.center_x,
        right.center_y - left.center_y,
    ) / scale
    size = abs(math.log(right.width / left.width)) + abs(
        math.log(right.height / left.height)
    )
    return center + 0.25 * size


def _box_scale(left: Detection, right: Detection) -> float:
    return max(
        4.0,
        math.sqrt(max(_EPS, left.width * left.height)),
        math.sqrt(max(_EPS, right.width * right.height)),
    )


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"similarity motion {name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"similarity motion {name} must be positive")
    return parsed


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"similarity motion {name} must be a finite scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"similarity motion {name} must be finite and non-negative"
        )
    return parsed


def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"similarity motion {name} must be positive")
    return parsed
