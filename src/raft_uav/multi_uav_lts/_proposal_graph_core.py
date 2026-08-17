"""Core graph construction for the Multi-UAV LTS proposal tracker."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import Detection, box_iou

_INVALID = 1e9
_EPS = 1e-12


class GraphParameters(Protocol):
    min_proposal_confidence: float
    duplicate_iou: float
    min_seed_iou: float
    anchor_max_cost: float
    anchor_min_margin: float
    enable_global_links: bool
    max_link_gap: int
    max_link_cost: float
    center_weight: float
    size_weight: float
    iou_weight: float
    velocity_weight: float
    gap_weight: float
    confidence_weight: float
    birth_min_hits: int
    birth_min_span: int
    birth_min_mean_confidence: float
    image_width: float | None
    image_height: float | None
    border_margin_fraction: float
    border_gap_discount: float


@dataclass(frozen=True)
class CoreResult:
    rows: tuple[Detection, ...]
    retained_rows: int
    suppressed_rows: int
    anchor_tracklets: int
    graph_links: int
    seeded_paths: int
    birth_paths: int
    dropped_paths: int


@dataclass(frozen=True)
class _Node:
    index: int
    row: Detection


@dataclass(frozen=True)
class _Tracklet:
    index: int
    rows: tuple[Detection, ...]
    node_indices: tuple[int, ...]
    seed_id: int | None = None

    @property
    def start(self) -> int:
        return self.rows[0].frame_id

    @property
    def end(self) -> int:
        return self.rows[-1].frame_id


def track_sequence(
    seeds: tuple[Detection, ...],
    proposals: tuple[Detection, ...],
    parameters: GraphParameters,
) -> CoreResult:
    retained, suppressed = _canonicalize(proposals, parameters)
    nodes = tuple(_Node(index, row) for index, row in enumerate(retained))
    tracklets = _anchor_tracklets(nodes, parameters)
    tracklets = _attach_seeds(tracklets, nodes, seeds, parameters.min_seed_iou)
    links = _global_links(tracklets, parameters)
    paths = _paths(tracklets, links)
    rows, seeded, births, dropped = _materialize(paths, seeds, parameters)
    return CoreResult(
        rows=rows,
        retained_rows=len(retained),
        suppressed_rows=suppressed,
        anchor_tracklets=len(tracklets),
        graph_links=len(links),
        seeded_paths=seeded,
        birth_paths=births,
        dropped_paths=dropped,
    )


def _canonicalize(
    proposals: Sequence[Detection],
    p: GraphParameters,
) -> tuple[tuple[Detection, ...], int]:
    frames: dict[int, list[Detection]] = {}
    for row in proposals:
        if row.confidence >= p.min_proposal_confidence:
            frames.setdefault(row.frame_id, []).append(row)
    kept: list[Detection] = []
    suppressed = len(proposals) - sum(len(rows) for rows in frames.values())
    for frame in sorted(frames):
        selected: list[Detection] = []
        candidates = sorted(
            frames[frame],
            key=lambda row: (
                -row.confidence,
                row.x1,
                row.y1,
                row.width,
                row.height,
                row.object_id,
            ),
        )
        for candidate in candidates:
            if any(box_iou(candidate, row) >= p.duplicate_iou for row in selected):
                suppressed += 1
            else:
                selected.append(candidate)
        kept.extend(sorted(selected, key=lambda row: row.object_id))
    return tuple(kept), suppressed


def _anchor_tracklets(
    nodes: tuple[_Node, ...],
    p: GraphParameters,
) -> tuple[_Tracklet, ...]:
    frames: dict[int, list[_Node]] = {}
    for node in nodes:
        frames.setdefault(node.row.frame_id, []).append(node)
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for frame in sorted(frames):
        left = tuple(frames[frame])
        right = tuple(frames.get(frame + 1, ()))
        if not left or not right:
            continue
        costs = np.asarray(
            [[_observation_cost(a.row, b.row, p) for b in right] for a in left]
        )
        forward = np.argmin(costs, axis=1)
        backward = np.argmin(costs, axis=0)
        forward_margin = _margins(costs)
        backward_margin = _margins(costs.T)
        for left_index, raw_right_index in enumerate(forward):
            right_index = int(raw_right_index)
            if int(backward[right_index]) != left_index:
                continue
            if costs[left_index, right_index] > p.anchor_max_cost:
                continue
            if forward_margin[left_index] < p.anchor_min_margin:
                continue
            if backward_margin[right_index] < p.anchor_min_margin:
                continue
            left_node = left[left_index].index
            right_node = right[right_index].index
            successor[left_node] = right_node
            predecessor[right_node] = left_node

    by_index = {node.index: node for node in nodes}
    visited: set[int] = set()
    result: list[_Tracklet] = []
    for node in nodes:
        if node.index in predecessor:
            continue
        chain: list[int] = []
        current = node.index
        while current not in visited:
            chain.append(current)
            visited.add(current)
            if current not in successor:
                break
            current = successor[current]
        result.append(
            _Tracklet(
                len(result),
                tuple(by_index[index].row for index in chain),
                tuple(chain),
            )
        )
    for node in nodes:
        if node.index not in visited:
            result.append(_Tracklet(len(result), (node.row,), (node.index,)))
    return tuple(result)


def _margins(costs: np.ndarray) -> np.ndarray:
    if costs.shape[1] <= 1:
        return np.full(costs.shape[0], np.inf)
    ordered = np.partition(costs, 1, axis=1)
    return ordered[:, 1] - ordered[:, 0]


def _observation_cost(left: Detection, right: Detection, p: GraphParameters) -> float:
    scale = _scale(left, right)
    center = math.hypot(
        right.center_x - left.center_x,
        right.center_y - left.center_y,
    ) / scale
    size = abs(math.log(right.width / left.width)) + abs(
        math.log(right.height / left.height)
    )
    return (
        p.center_weight * center
        + p.size_weight * size
        + p.iou_weight * (1.0 - box_iou(left, right))
        + p.confidence_weight * (1.0 - right.confidence)
    )


def _scale(left: Detection, right: Detection) -> float:
    return max(
        4.0,
        math.sqrt(max(_EPS, left.width * left.height)),
        math.sqrt(max(_EPS, right.width * right.height)),
    )


def _attach_seeds(
    tracklets: tuple[_Tracklet, ...],
    nodes: tuple[_Node, ...],
    seeds: tuple[Detection, ...],
    min_iou: float,
) -> tuple[_Tracklet, ...]:
    node_to_tracklet = {
        node_index: tracklet.index
        for tracklet in tracklets
        for node_index in tracklet.node_indices
    }
    frame_one = tuple(node for node in nodes if node.row.frame_id == 1)
    mapping = _seed_mapping(seeds, frame_one, min_iou)
    seed_rows = {row.object_id: row for row in seeds}
    by_tracklet = {
        node_to_tracklet[node_index]: seed_id
        for node_index, seed_id in mapping.items()
    }
    result: list[_Tracklet] = []
    for tracklet in tracklets:
        seed_id = by_tracklet.get(tracklet.index)
        rows = tracklet.rows
        if seed_id is not None:
            exact = seed_rows[seed_id]
            rows = tuple(exact if row.frame_id == 1 else row for row in rows)
        result.append(
            _Tracklet(len(result), rows, tracklet.node_indices, seed_id=seed_id)
        )
    matched = set(mapping.values())
    for seed in seeds:
        if seed.object_id not in matched:
            result.append(_Tracklet(len(result), (seed,), (), seed_id=seed.object_id))
    return tuple(result)


def _seed_mapping(
    seeds: tuple[Detection, ...],
    nodes: tuple[_Node, ...],
    min_iou: float,
) -> dict[int, int]:
    if not seeds or not nodes:
        return {}
    overlaps = np.asarray(
        [[box_iou(seed, node.row) for node in nodes] for seed in seeds]
    )
    valid = (overlaps > 0.0) & (overlaps >= min_iou)
    bonus = float(min(overlaps.shape) + 1)
    rows, cols = linear_sum_assignment(-np.where(valid, bonus + overlaps, 0.0))
    return {
        nodes[node_index].index: seeds[seed_index].object_id
        for seed_index, node_index in zip(rows, cols, strict=True)
        if valid[seed_index, node_index]
    }


def _global_links(
    tracklets: tuple[_Tracklet, ...],
    p: GraphParameters,
) -> dict[int, int]:
    if not p.enable_global_links or len(tracklets) < 2:
        return {}
    candidates = _candidate_links(tracklets, p)
    if not candidates:
        return {}
    components = _link_components(candidates)
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
        links.update(_solve_link_component(component_candidates, p.max_link_cost))
    return links


def _candidate_links(
    tracklets: tuple[_Tracklet, ...],
    p: GraphParameters,
) -> dict[tuple[int, int], float]:
    starts: dict[int, list[_Tracklet]] = {}
    for tracklet in tracklets:
        starts.setdefault(tracklet.start, []).append(tracklet)
    frames = sorted(starts)
    candidates: dict[tuple[int, int], float] = {}
    for left in tracklets:
        lower = bisect_left(frames, left.end + 1)
        upper = bisect_right(frames, left.end + p.max_link_gap + 1)
        for frame in frames[lower:upper]:
            for right in starts[frame]:
                cost = _link_cost(left, right, p)
                if math.isfinite(cost) and cost < p.max_link_cost:
                    candidates[(left.index, right.index)] = cost
    return candidates


def _link_components(
    candidates: dict[tuple[int, int], float],
) -> tuple[tuple[int, ...], ...]:
    adjacency: dict[int, set[int]] = {}
    for left, right in candidates:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for root in sorted(adjacency):
        if root in visited:
            continue
        stack = [root]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited, reverse=True))
        components.append(tuple(sorted(component)))
    return tuple(components)


def _solve_link_component(
    candidates: dict[tuple[int, int], float],
    max_link_cost: float,
) -> dict[int, int]:
    left_ids = sorted({left for left, _right in candidates})
    right_ids = sorted({right for _left, right in candidates})
    if not left_ids or not right_ids:
        return {}
    left_position = {value: index for index, value in enumerate(left_ids)}
    right_position = {value: index for index, value in enumerate(right_ids)}
    left_count = len(left_ids)
    right_count = len(right_ids)
    size = left_count + right_count
    matrix = np.full((size, size), _INVALID)
    for (left, right), cost in candidates.items():
        if left in left_position and right in right_position:
            matrix[left_position[left], right_position[right]] = cost
    unmatched = 0.5 * max_link_cost
    for index in range(left_count):
        matrix[index, right_count + index] = unmatched
    for index in range(right_count):
        matrix[left_count + index, index] = unmatched
    matrix[left_count:, right_count:] = 0.0
    rows, cols = linear_sum_assignment(matrix)
    links: dict[int, int] = {}
    for row, col in zip(rows, cols, strict=True):
        if row >= left_count or col >= right_count:
            continue
        left = left_ids[int(row)]
        right = right_ids[int(col)]
        if candidates.get((left, right), math.inf) < max_link_cost:
            links[left] = right
    return links


def _link_cost(left: _Tracklet, right: _Tracklet, p: GraphParameters) -> float:
    if left.index == right.index or left.end >= right.start:
        return math.inf
    gap = right.start - left.end - 1
    if gap > p.max_link_gap:
        return math.inf
    predicted = _predict(left.rows, right.start)
    first = right.rows[0]
    scale = _scale(predicted, first)
    center = math.hypot(
        first.center_x - predicted.center_x,
        first.center_y - predicted.center_y,
    ) / scale
    size = abs(math.log(first.width / predicted.width)) + abs(
        math.log(first.height / predicted.height)
    )
    motion = p.center_weight * center + p.gap_weight * gap
    if gap > 0 and _border_pair(left.rows[-1], first, p):
        motion *= p.border_gap_discount
    return (
        motion
        + p.size_weight * size
        + p.iou_weight * (1.0 - box_iou(predicted, first))
        + p.velocity_weight * _velocity_cost(left.rows, right.rows, scale)
        + p.confidence_weight * (1.0 - first.confidence)
    )


def _predict(rows: tuple[Detection, ...], frame: int) -> Detection:
    last = rows[-1]
    if len(rows) < 2 or frame <= last.frame_id:
        return replace(last, frame_id=frame)
    previous = rows[-2]
    history = last.frame_id - previous.frame_id
    if history <= 0:
        return replace(last, frame_id=frame)
    delta = frame - last.frame_id
    vx = (last.center_x - previous.center_x) / history
    vy = (last.center_y - previous.center_y) / history
    return replace(
        last,
        frame_id=frame,
        x1=last.center_x + vx * delta - 0.5 * last.width,
        y1=last.center_y + vy * delta - 0.5 * last.height,
    )


def _velocity_cost(
    left: tuple[Detection, ...],
    right: tuple[Detection, ...],
    scale: float,
) -> float:
    a = _velocity(left, tail=True)
    b = _velocity(right, tail=False)
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1]) / scale


def _velocity(
    rows: tuple[Detection, ...],
    *,
    tail: bool,
) -> tuple[float, float] | None:
    if len(rows) < 2:
        return None
    left, right = (rows[-2], rows[-1]) if tail else (rows[0], rows[1])
    delta = right.frame_id - left.frame_id
    if delta <= 0:
        return None
    return (
        (right.center_x - left.center_x) / delta,
        (right.center_y - left.center_y) / delta,
    )


def _border_pair(left: Detection, right: Detection, p: GraphParameters) -> bool:
    if p.image_width is None or p.image_height is None:
        return False
    margin_x = p.image_width * p.border_margin_fraction
    margin_y = p.image_height * p.border_margin_fraction
    return _near_border(left, p.image_width, p.image_height, margin_x, margin_y) and (
        _near_border(right, p.image_width, p.image_height, margin_x, margin_y)
    )


def _near_border(
    row: Detection,
    width: float,
    height: float,
    margin_x: float,
    margin_y: float,
) -> bool:
    return (
        row.x1 <= margin_x
        or row.y1 <= margin_y
        or width - row.x1 - row.width <= margin_x
        or height - row.y1 - row.height <= margin_y
    )


def _paths(
    tracklets: tuple[_Tracklet, ...],
    links: dict[int, int],
) -> tuple[tuple[_Tracklet, ...], ...]:
    predecessor = {right: left for left, right in links.items()}
    visited: set[int] = set()
    result: list[tuple[_Tracklet, ...]] = []
    for tracklet in tracklets:
        if tracklet.index in predecessor:
            continue
        chain: list[_Tracklet] = []
        current = tracklet.index
        while current not in visited:
            chain.append(tracklets[current])
            visited.add(current)
            if current not in links:
                break
            current = links[current]
        result.append(tuple(chain))
    result.extend((row,) for row in tracklets if row.index not in visited)
    return tuple(result)


def _materialize(
    paths: tuple[tuple[_Tracklet, ...], ...],
    seeds: tuple[Detection, ...],
    p: GraphParameters,
) -> tuple[tuple[Detection, ...], int, int, int]:
    seeded: list[tuple[int, tuple[Detection, ...]]] = []
    births: list[tuple[tuple[Detection, ...], tuple[float, ...]]] = []
    dropped = 0
    for path in paths:
        rows = tuple(
            sorted(
                (row for tracklet in path for row in tracklet.rows),
                key=lambda row: row.frame_id,
            )
        )
        seed_ids = {item.seed_id for item in path if item.seed_id is not None}
        if len(seed_ids) > 1:
            raise RuntimeError("proposal graph joined two seeded identities")
        if seed_ids:
            seeded.append((next(iter(seed_ids)), rows))
            continue
        span = rows[-1].frame_id - rows[0].frame_id
        mean_confidence = float(np.mean([row.confidence for row in rows]))
        if (
            len(rows) >= p.birth_min_hits
            and span >= p.birth_min_span
            and mean_confidence >= p.birth_min_mean_confidence
        ):
            key = (
                float(rows[0].frame_id),
                rows[0].center_x,
                rows[0].center_y,
                -mean_confidence,
            )
            births.append((rows, key))
        else:
            dropped += 1
    output: list[Detection] = []
    for object_id, rows in sorted(seeded, key=lambda item: item[0]):
        output.extend(replace(row, object_id=object_id) for row in rows)
    next_id = max((row.object_id for row in seeds), default=0) + 1
    for rows, _key in sorted(births, key=lambda item: item[1]):
        output.extend(replace(row, object_id=next_id) for row in rows)
        next_id += 1
    output.sort(key=lambda row: (row.frame_id, row.object_id))
    return tuple(output), len(seeded), len(births), dropped
