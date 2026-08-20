"""Delayed global path-cover association for Multi-UAV LTS proposal graphs.

The existing proposal graph forms irreversible reciprocal nearest-neighbour
anchors before it reasons over longer gaps. This experimental path keeps every
canonical proposal atomic during the short-horizon association stage. Link
costs can use bounded future-continuation evidence, a local swarm-geometry
penalty, and a calibrated same-identity edge likelihood. All short links are
then selected jointly by the existing sparse optional-assignment solver.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from . import _proposal_graph_core as core
from ._proposal_edge_likelihood import (
    SWARM_RELATIVE_FEATURE_INDEX,
    EdgeFeatureContext,
    EdgeLikelihoodModel,
    build_edge_feature_context,
    edge_feature_vector,
)
from ._records import Detection, box_iou


@dataclass(frozen=True)
class DelayedPathCoverConfig:
    """Controls for delayed association and optional edge rescoring."""

    max_gap: int = 0
    lookahead_frames: int = 2
    successors_per_frame: int = 3
    continuation_weight: float = 0.75
    continuation_clip: float = 4.0
    edge_model_weight: float = 1.0
    edge_model_clip: float = 4.0
    swarm_relative_weight: float = 0.0
    swarm_relative_clip: float = 4.0
    swarm_neighbors: int = 4
    swarm_radius_scale: float = 12.0
    swarm_unmatched_penalty: float = 2.0

    def validate(self) -> None:
        _nonnegative_int(self.max_gap, name="max_gap")
        _positive_int(self.lookahead_frames, name="lookahead_frames")
        _positive_int(self.successors_per_frame, name="successors_per_frame")
        _nonnegative_finite(self.continuation_weight, name="continuation_weight")
        _positive_finite(self.continuation_clip, name="continuation_clip")
        _nonnegative_finite(self.edge_model_weight, name="edge_model_weight")
        _positive_finite(self.edge_model_clip, name="edge_model_clip")
        _nonnegative_finite(
            self.swarm_relative_weight,
            name="swarm_relative_weight",
        )
        _positive_finite(self.swarm_relative_clip, name="swarm_relative_clip")
        _positive_int(self.swarm_neighbors, name="swarm_neighbors")
        _positive_finite(self.swarm_radius_scale, name="swarm_radius_scale")
        _positive_finite(
            self.swarm_unmatched_penalty,
            name="swarm_unmatched_penalty",
        )


def track_sequence_delayed_path_cover(
    seeds: tuple[Detection, ...],
    proposals: tuple[Detection, ...],
    parameters: core.GraphParameters,
    *,
    config: DelayedPathCoverConfig,
    edge_model: EdgeLikelihoodModel | None = None,
) -> core.CoreResult:
    """Track one sequence without irreversible frame-local anchor decisions."""

    config.validate()
    if edge_model is not None:
        edge_model.validate()
    retained, suppressed = core._canonicalize(proposals, parameters)
    nodes = tuple(core._Node(index, row) for index, row in enumerate(retained))
    common_motion = core._estimate_common_motion(nodes, parameters)

    # Keep every proposal atomic until the delayed path-cover solve. Exact frame-one
    # seed boxes replace matched proposal boxes through the existing seed attachment.
    atomic = tuple(
        core._Tracklet(index, (node.row,), (node.index,))
        for index, node in enumerate(nodes)
    )
    atomic = core._attach_seeds(
        atomic,
        nodes,
        seeds,
        parameters.min_seed_iou,
    )
    feature_context = _feature_context(atomic, config, edge_model)

    short_candidates = _delayed_candidates(
        atomic,
        parameters,
        common_motion,
        config,
        edge_model=edge_model,
        feature_context=feature_context,
    )
    short_links = _solve_components(short_candidates, parameters.max_link_cost)
    short_paths = core._paths(atomic, short_links)
    micro_tracklets = _collapse_paths(short_paths)

    # Preserve the mature long-gap machinery: now it operates on trajectories whose
    # short ambiguous links were selected with delayed and swarm-relative evidence.
    long_links = core._global_links(micro_tracklets, parameters, common_motion)
    final_paths = core._paths(micro_tracklets, long_links)
    rows, seeded, births, dropped, interpolated = core._materialize(
        final_paths,
        seeds,
        parameters,
    )
    return core.CoreResult(
        rows=rows,
        retained_rows=len(retained),
        suppressed_rows=suppressed,
        anchor_tracklets=len(micro_tracklets),
        graph_links=len(short_links) + len(long_links),
        common_motion_steps=len(common_motion),
        interpolated_rows=interpolated,
        seeded_paths=seeded,
        birth_paths=births,
        dropped_paths=dropped,
    )


def _feature_context(
    tracklets: tuple[core._Tracklet, ...],
    config: DelayedPathCoverConfig,
    edge_model: EdgeLikelihoodModel | None,
) -> EdgeFeatureContext | None:
    if edge_model is None and config.swarm_relative_weight <= 0.0:
        return None
    rows = tuple(row for tracklet in tracklets for row in tracklet.rows)
    return build_edge_feature_context(
        rows,
        neighbor_count=config.swarm_neighbors,
        radius_scale=config.swarm_radius_scale,
        unmatched_penalty=config.swarm_unmatched_penalty,
    )


def _delayed_candidates(
    tracklets: tuple[core._Tracklet, ...],
    parameters: core.GraphParameters,
    common_motion: dict[int, tuple[float, float]],
    config: DelayedPathCoverConfig,
    *,
    edge_model: EdgeLikelihoodModel | None,
    feature_context: EdgeFeatureContext | None,
) -> dict[tuple[int, int], float]:
    starts: dict[int, list[core._Tracklet]] = {}
    for tracklet in tracklets:
        starts.setdefault(tracklet.start, []).append(tracklet)
    frames = sorted(starts)
    candidates: dict[tuple[int, int], float] = {}

    for left in tracklets:
        lower = bisect_left(frames, left.end + 1)
        upper = bisect_right(frames, left.end + config.max_gap + 1)
        for frame in frames[lower:upper]:
            scored: list[
                tuple[float, int, core._Tracklet, float, float]
            ] = []
            for right in starts[frame]:
                raw = core._link_cost(left, right, parameters, common_motion)
                if not math.isfinite(raw) or raw >= parameters.max_link_cost:
                    continue
                model_cost = 0.0
                swarm_cost = 0.0
                if feature_context is not None:
                    predicted = core._predict(left.rows, right.start, common_motion)
                    features = edge_feature_vector(
                        left.rows[-1],
                        right.rows[0],
                        predicted,
                        gap_frames=right.start - left.end - 1,
                        context=feature_context,
                    )
                    swarm_cost = float(features[SWARM_RELATIVE_FEATURE_INDEX])
                    if edge_model is not None:
                        model_cost = edge_model.negative_log_probability(features)
                scored.append(
                    (raw, right.index, right, model_cost, swarm_cost)
                )
            if not scored:
                continue

            best_model_cost = min(item[3] for item in scored)
            best_swarm_cost = min(item[4] for item in scored)
            ranked: list[tuple[float, int, core._Tracklet]] = []
            for raw, right_index, right, model_cost, swarm_cost in scored:
                adjusted = raw
                if edge_model is not None and config.edge_model_weight > 0.0:
                    relative_model = max(0.0, model_cost - best_model_cost)
                    adjusted += config.edge_model_weight * min(
                        config.edge_model_clip,
                        relative_model,
                    )
                if config.swarm_relative_weight > 0.0:
                    relative_swarm = max(0.0, swarm_cost - best_swarm_cost)
                    adjusted += config.swarm_relative_weight * min(
                        config.swarm_relative_clip,
                        relative_swarm,
                    )
                if adjusted < parameters.max_link_cost:
                    ranked.append((adjusted, right_index, right))
            ranked.sort(key=lambda item: (item[0], item[1]))

            delayed: list[tuple[float, int, core._Tracklet, float]] = []
            for prescore, right_index, right in ranked[: config.successors_per_frame]:
                continuation = _continuation_potential(
                    left,
                    right,
                    starts,
                    frames,
                    parameters,
                    common_motion,
                    config.lookahead_frames,
                )
                delayed.append((prescore, right_index, right, continuation))
            if not delayed:
                continue
            best_continuation = min(item[3] for item in delayed)
            for prescore, _right_index, right, continuation in delayed:
                relative = max(0.0, continuation - best_continuation)
                adjusted = prescore + config.continuation_weight * min(
                    config.continuation_clip,
                    relative,
                )
                if adjusted < parameters.max_link_cost:
                    candidates[(left.index, right.index)] = adjusted
    return candidates


def _continuation_potential(
    left: core._Tracklet,
    right: core._Tracklet,
    starts: dict[int, list[core._Tracklet]],
    frames: Sequence[int],
    parameters: core.GraphParameters,
    common_motion: dict[int, tuple[float, float]],
    horizon: int,
) -> float:
    """Score how well a proposed link admits a constant-residual-motion future."""

    first = left.rows[-1]
    second = right.rows[0]
    delta = second.frame_id - first.frame_id
    if delta <= 0:
        return 0.0
    common_dx, common_dy = core._motion_between(
        common_motion,
        first.frame_id,
        second.frame_id,
    )
    residual_vx = (second.center_x - first.center_x - common_dx) / delta
    residual_vy = (second.center_y - first.center_y - common_dy) / delta

    lower = bisect_left(frames, second.frame_id + 1)
    upper = bisect_right(frames, second.frame_id + horizon)
    per_frame: list[float] = []
    for frame in frames[lower:upper]:
        future = starts.get(frame, ())
        if not future:
            continue
        best = min(
            _future_cost(
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
    if not per_frame:
        return 0.0
    return float(np.mean(per_frame))


def _future_cost(
    current: Detection,
    future: Detection,
    residual_vx: float,
    residual_vy: float,
    parameters: core.GraphParameters,
    common_motion: dict[int, tuple[float, float]],
) -> float:
    delta = future.frame_id - current.frame_id
    if delta <= 0:
        return math.inf
    common_dx, common_dy = core._motion_between(
        common_motion,
        current.frame_id,
        future.frame_id,
    )
    predicted_center_x = current.center_x + common_dx + residual_vx * delta
    predicted_center_y = current.center_y + common_dy + residual_vy * delta
    predicted = replace(
        current,
        frame_id=future.frame_id,
        x1=predicted_center_x - 0.5 * current.width,
        y1=predicted_center_y - 0.5 * current.height,
    )
    scale = core._scale(predicted, future)
    center = math.hypot(
        future.center_x - predicted.center_x,
        future.center_y - predicted.center_y,
    ) / scale
    size = abs(math.log(future.width / predicted.width)) + abs(
        math.log(future.height / predicted.height)
    )
    next_common_dx, next_common_dy = core._motion_between(
        common_motion,
        current.frame_id,
        future.frame_id,
    )
    next_vx = (future.center_x - current.center_x - next_common_dx) / delta
    next_vy = (future.center_y - current.center_y - next_common_dy) / delta
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


def _solve_components(
    candidates: dict[tuple[int, int], float],
    max_link_cost: float,
) -> dict[int, int]:
    if not candidates:
        return {}
    components = core._link_components(candidates)
    component_index = {
        node: index
        for index, component in enumerate(components)
        for node in component
    }
    grouped: list[dict[tuple[int, int], float]] = [dict() for _ in components]
    for edge, cost in candidates.items():
        grouped[component_index[edge[0]]][edge] = cost
    links: dict[int, int] = {}
    for component_candidates in grouped:
        links.update(core._solve_link_component(component_candidates, max_link_cost))
    return links


def _collapse_paths(
    paths: tuple[tuple[core._Tracklet, ...], ...],
) -> tuple[core._Tracklet, ...]:
    collapsed: list[core._Tracklet] = []
    for path in paths:
        seed_ids = {item.seed_id for item in path if item.seed_id is not None}
        if len(seed_ids) > 1:
            raise RuntimeError("delayed path cover joined two seeded identities")
        rows = tuple(
            sorted(
                (row for tracklet in path for row in tracklet.rows),
                key=frame_object_key,
            )
        )
        node_indices = tuple(
            index for tracklet in path for index in tracklet.node_indices
        )
        collapsed.append(
            core._Tracklet(
                len(collapsed),
                rows,
                node_indices,
                seed_id=next(iter(seed_ids)) if seed_ids else None,
            )
        )
    return tuple(collapsed)


def frame_object_key(row: Detection) -> tuple[int, int]:
    return row.frame_id, row.object_id


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"delayed {name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    parsed = _nonnegative_int(value, name=name)
    if parsed <= 0:
        raise ValueError(f"delayed {name} must be positive")
    return parsed


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"delayed {name} must be a finite scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"delayed {name} must be finite and non-negative")
    return parsed


def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"delayed {name} must be positive")
    return parsed
