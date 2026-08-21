"""Ambiguity-triggered beam reranking for delayed proposal path cover.

The ordinary delayed path cover commits to the minimum pairwise-link assignment.
This module retains a bounded set of close one-to-one assignments for small
ambiguous components and reranks them with a second-order residual-motion score.
Easy or large components use the existing exact optional-matching solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol

from . import _proposal_delayed_path_cover as delayed
from . import _proposal_graph_core as core
from ._proposal_edge_likelihood import EdgeLikelihoodModel
from ._records import Detection


class _Row(Protocol):
    frame_id: int
    center_x: float
    center_y: float
    width: float
    height: float


class _Tracklet(Protocol):
    index: int
    rows: tuple[_Row, ...]
    seed_id: int | None


@dataclass(frozen=True)
class AmbiguityBeamConfig:
    """Controls for bounded alternative-assignment reranking."""

    width: int = 8
    max_component_nodes: int = 16
    margin: float = 1.0
    acceleration_weight: float = 1.0
    acceleration_clip: float = 4.0
    expansion_factor: int = 4

    def validate(self) -> None:
        _positive_int(self.width, name="width")
        _positive_int(self.max_component_nodes, name="max_component_nodes")
        _nonnegative_finite(self.margin, name="margin")
        _nonnegative_finite(
            self.acceleration_weight,
            name="acceleration_weight",
        )
        _positive_finite(self.acceleration_clip, name="acceleration_clip")
        _positive_int(self.expansion_factor, name="expansion_factor")


def track_sequence_ambiguity_beam(
    seeds: tuple[Detection, ...],
    proposals: tuple[Detection, ...],
    parameters: core.GraphParameters,
    *,
    delayed_config: delayed.DelayedPathCoverConfig,
    beam_config: AmbiguityBeamConfig,
    edge_model: EdgeLikelihoodModel | None = None,
) -> core.CoreResult:
    """Track one sequence with bounded alternative short-link hypotheses."""

    delayed_config.validate()
    beam_config.validate()
    if edge_model is not None:
        edge_model.validate()
    retained, suppressed = core._canonicalize(proposals, parameters)
    nodes = tuple(core._Node(index, row) for index, row in enumerate(retained))
    common_motion = core._estimate_common_motion(nodes, parameters)
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
    feature_context = delayed._feature_context(
        atomic,
        delayed_config,
        edge_model,
    )
    short_candidates = delayed._delayed_candidates(
        atomic,
        parameters,
        common_motion,
        delayed_config,
        edge_model=edge_model,
        feature_context=feature_context,
    )
    short_links = _solve_components(
        short_candidates,
        parameters.max_link_cost,
        atomic,
        common_motion,
        config=beam_config,
    )
    short_paths = core._paths(atomic, short_links)
    micro_tracklets = delayed._collapse_paths(short_paths)
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


def _solve_components(
    candidates: dict[tuple[int, int], float],
    max_link_cost: float,
    tracklets: tuple[core._Tracklet, ...],
    common_motion: dict[int, tuple[float, float]],
    *,
    config: AmbiguityBeamConfig,
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
    by_index = {tracklet.index: tracklet for tracklet in tracklets}
    links: dict[int, int] = {}
    for component_candidates in grouped:
        exact = core._solve_link_component(component_candidates, max_link_cost)
        links.update(
            solve_component(
                component_candidates,
                max_link_cost,
                by_index,
                common_motion,
                config=config,
                exact_links=exact,
            )
        )
    return links


@dataclass(frozen=True)
class _Hypothesis:
    adjusted_cost: float
    used_rights: frozenset[int]
    links: tuple[tuple[int, int], ...]


def solve_component(
    candidates: Mapping[tuple[int, int], float],
    max_link_cost: float,
    tracklets: Mapping[int, _Tracklet],
    common_motion: Mapping[int, tuple[float, float]],
    *,
    config: AmbiguityBeamConfig,
    exact_links: Mapping[int, int],
) -> dict[int, int]:
    """Rerank close optional matchings with second-order motion evidence.

    ``exact_links`` must be the result of the maintained pairwise solver. It is
    always retained as a candidate and is returned unchanged when the component
    is large, unambiguous, or the higher-order score is disabled.
    """

    config.validate()
    if not candidates:
        return {}
    left_ids = tuple(sorted({left for left, _right in candidates}))
    right_ids = tuple(sorted({right for _left, right in candidates}))
    component_nodes = set(left_ids) | set(right_ids)
    if (
        len(component_nodes) > config.max_component_nodes
        or config.width <= 1
        or config.acceleration_weight <= 0.0
    ):
        return dict(exact_links)

    hypotheses = _beam_hypotheses(
        candidates,
        left_ids,
        right_ids,
        max_link_cost=max_link_cost,
        width=config.width,
        expansion_factor=config.expansion_factor,
    )
    exact_tuple = tuple(
        sorted((int(left), int(right)) for left, right in exact_links.items())
    )
    all_links = {hypothesis.links: hypothesis for hypothesis in hypotheses}
    if exact_tuple not in all_links:
        all_links[exact_tuple] = _Hypothesis(
            adjusted_cost=_adjusted_cost(exact_tuple, candidates, max_link_cost),
            used_rights=frozenset(right for _left, right in exact_tuple),
            links=exact_tuple,
        )
    ranked = sorted(
        all_links.values(),
        key=lambda hypothesis: (
            _base_cost(
                hypothesis.adjusted_cost,
                len(left_ids),
                len(right_ids),
                max_link_cost,
            ),
            hypothesis.links,
        ),
    )[: config.width]
    if len(ranked) <= 1:
        return dict(ranked[0].links) if ranked else dict(exact_links)

    best_base = _base_cost(
        ranked[0].adjusted_cost,
        len(left_ids),
        len(right_ids),
        max_link_cost,
    )
    second_base = _base_cost(
        ranked[1].adjusted_cost,
        len(left_ids),
        len(right_ids),
        max_link_cost,
    )
    if second_base - best_base > config.margin:
        return dict(exact_links)

    rescored: list[tuple[float, float, tuple[tuple[int, int], ...]]] = []
    for hypothesis in ranked:
        base = _base_cost(
            hypothesis.adjusted_cost,
            len(left_ids),
            len(right_ids),
            max_link_cost,
        )
        acceleration = _acceleration_penalty(
            hypothesis.links,
            component_nodes,
            tracklets,
            common_motion,
            clip=config.acceleration_clip,
        )
        total = base + config.acceleration_weight * acceleration
        rescored.append((total, base, hypothesis.links))
    rescored.sort(key=lambda item: (item[0], item[1], item[2]))
    return dict(rescored[0][2])


def _beam_hypotheses(
    candidates: Mapping[tuple[int, int], float],
    left_ids: tuple[int, ...],
    right_ids: tuple[int, ...],
    *,
    max_link_cost: float,
    width: int,
    expansion_factor: int,
) -> tuple[_Hypothesis, ...]:
    outgoing: dict[int, tuple[tuple[int, float], ...]] = {}
    for left in left_ids:
        outgoing[left] = tuple(
            sorted(
                (
                    (right, float(cost) - max_link_cost)
                    for (edge_left, right), cost in candidates.items()
                    if edge_left == left
                    and math.isfinite(float(cost))
                    and float(cost) < max_link_cost
                ),
                key=lambda item: (item[1], item[0]),
            )
        )

    beam = (_Hypothesis(0.0, frozenset(), ()),)
    keep = max(width, width * expansion_factor)
    for left in left_ids:
        expanded: list[_Hypothesis] = []
        for hypothesis in beam:
            expanded.append(hypothesis)
            for right, adjustment in outgoing[left]:
                if right in hypothesis.used_rights:
                    continue
                expanded.append(
                    _Hypothesis(
                        adjusted_cost=hypothesis.adjusted_cost + adjustment,
                        used_rights=hypothesis.used_rights | {right},
                        links=hypothesis.links + ((left, right),),
                    )
                )
        expanded.sort(
            key=lambda hypothesis: (
                hypothesis.adjusted_cost,
                hypothesis.links,
            )
        )
        deduplicated: list[_Hypothesis] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        for hypothesis in expanded:
            if hypothesis.links in seen:
                continue
            seen.add(hypothesis.links)
            deduplicated.append(hypothesis)
            if len(deduplicated) >= keep:
                break
        beam = tuple(deduplicated)
    return tuple(
        sorted(
            beam,
            key=lambda hypothesis: (hypothesis.adjusted_cost, hypothesis.links),
        )[:width]
    )


def _adjusted_cost(
    links: tuple[tuple[int, int], ...],
    candidates: Mapping[tuple[int, int], float],
    max_link_cost: float,
) -> float:
    return float(sum(float(candidates[edge]) - max_link_cost for edge in links))


def _base_cost(
    adjusted_cost: float,
    left_count: int,
    right_count: int,
    max_link_cost: float,
) -> float:
    unmatched_side = 0.5 * max_link_cost
    return unmatched_side * (left_count + right_count) + adjusted_cost


def _acceleration_penalty(
    links: tuple[tuple[int, int], ...],
    component_nodes: set[int],
    tracklets: Mapping[int, _Tracklet],
    common_motion: Mapping[int, tuple[float, float]],
    *,
    clip: float,
) -> float:
    successor = dict(links)
    predecessor = {right: left for left, right in links}
    penalty = 0.0
    visited: set[int] = set()
    for start in sorted(component_nodes):
        if start in predecessor:
            continue
        path: list[int] = []
        current = start
        while current not in visited:
            path.append(current)
            visited.add(current)
            if current not in successor:
                break
            current = successor[current]
        penalty += _path_acceleration(
            path,
            tracklets,
            common_motion,
            clip=clip,
        )
    for start in sorted(component_nodes - visited):
        penalty += _path_acceleration(
            [start],
            tracklets,
            common_motion,
            clip=clip,
        )
    return penalty


def _path_acceleration(
    path: list[int],
    tracklets: Mapping[int, _Tracklet],
    common_motion: Mapping[int, tuple[float, float]],
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
    anchor = rows[0].frame_id
    residual_positions: list[tuple[float, float]] = []
    for row in rows:
        dx, dy = _motion_between(common_motion, anchor, row.frame_id)
        residual_positions.append((row.center_x - dx, row.center_y - dy))
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
        scale = max(
            4.0,
            math.sqrt(max(1e-12, middle.width * middle.height)),
        )
        normalized = math.hypot(
            right_velocity[0] - left_velocity[0],
            right_velocity[1] - left_velocity[1],
        ) / scale
        result += min(clip, normalized)
    return result


def _motion_between(
    common_motion: Mapping[int, tuple[float, float]],
    start_frame: int,
    end_frame: int,
) -> tuple[float, float]:
    dx = 0.0
    dy = 0.0
    for frame in range(start_frame, end_frame):
        step_dx, step_dy = common_motion.get(frame, (0.0, 0.0))
        dx += float(step_dx)
        dy += float(step_dy)
    return dx, dy


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"ambiguity beam {name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    parsed = _nonnegative_int(value, name=name)
    if parsed <= 0:
        raise ValueError(f"ambiguity beam {name} must be positive")
    return parsed


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ambiguity beam {name} must be a finite scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"ambiguity beam {name} must be finite and non-negative"
        )
    return parsed


def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"ambiguity beam {name} must be positive")
    return parsed
