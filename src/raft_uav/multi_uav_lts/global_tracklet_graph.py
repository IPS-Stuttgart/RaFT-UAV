"""Offline variable-cardinality tracklet graph for Multi-UAV LTS proposals.

The tracker converts permissive per-frame detector proposals into short,
high-confidence tracklets, links those tracklets with one sparse global
assignment, preserves benchmark-supplied frame-one identities, and optionally
confirms persistent late births.  It is an experiment candidate rather than an
oracle: no truth rows are consumed during tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from ._records import (
    Detection,
    box_iou,
    format_detection,
    iou_matrix,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)

_EPS = 1e-12
_INVALID_COST = 1e9


@dataclass(frozen=True)
class GlobalTrackletGraphParameters:
    """Validated controls for the offline proposal graph."""

    min_confidence: float = 0.003
    nms_iou: float = 0.95
    reciprocal_local_links: bool = True
    local_max_normalized_distance: float = 3.0
    local_max_log_size_change: float = 1.25
    local_max_cost: float = 2.25
    local_min_margin: float = 0.05
    global_max_gap: int = 12
    max_global_normalized_distance: float = 6.0
    max_global_log_size_change: float = 1.75
    max_global_velocity_mismatch: float = 4.0
    max_link_cost: float = 4.5
    birth_cost: float = 2.5
    death_cost: float = 2.5
    edge_limit_per_tracklet: int = 8
    min_link_tracklet_length: int = 2
    allow_births: bool = True
    allow_frame_one_unseeded_births: bool = False
    min_birth_frames: int = 3
    min_birth_span: int = 2
    min_birth_mean_confidence: float = 0.01
    seed_min_iou: float = 0.25
    enable_border_reentry: bool = False
    border_max_gap: int = 90
    border_margin_px: float = 32.0
    border_link_penalty: float = 0.75
    frame_width: float | None = None
    frame_height: float | None = None

    def __post_init__(self) -> None:
        unit_fields = (
            "min_confidence",
            "nms_iou",
            "min_birth_mean_confidence",
            "seed_min_iou",
        )
        finite_fields = (
            "local_max_normalized_distance",
            "local_max_log_size_change",
            "local_max_cost",
            "local_min_margin",
            "max_global_normalized_distance",
            "max_global_log_size_change",
            "max_global_velocity_mismatch",
            "max_link_cost",
            "birth_cost",
            "death_cost",
            "border_margin_px",
            "border_link_penalty",
        )
        integer_fields = (
            "global_max_gap",
            "edge_limit_per_tracklet",
            "min_link_tracklet_length",
            "min_birth_frames",
            "min_birth_span",
            "border_max_gap",
        )
        boolean_fields = (
            "reciprocal_local_links",
            "allow_births",
            "allow_frame_one_unseeded_births",
            "enable_border_reentry",
        )
        for name in unit_fields:
            object.__setattr__(
                self,
                name,
                validate_unit_interval(getattr(self, name), name=name),
            )
        for name in finite_fields:
            object.__setattr__(
                self,
                name,
                validate_nonnegative_finite(getattr(self, name), name=name),
            )
        for name in integer_fields:
            object.__setattr__(
                self,
                name,
                validate_nonnegative_int(getattr(self, name), name=name),
            )
        for name in boolean_fields:
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name=name))
        if self.edge_limit_per_tracklet <= 0:
            raise ValueError("edge_limit_per_tracklet must be positive")
        if self.min_link_tracklet_length <= 0:
            raise ValueError("min_link_tracklet_length must be positive")
        if self.min_birth_frames <= 0:
            raise ValueError("min_birth_frames must be positive")
        if self.border_max_gap < self.global_max_gap:
            raise ValueError("border_max_gap must be at least global_max_gap")
        width = _optional_positive_finite(self.frame_width, name="frame_width")
        height = _optional_positive_finite(self.frame_height, name="frame_height")
        object.__setattr__(self, "frame_width", width)
        object.__setattr__(self, "frame_height", height)
        if self.enable_border_reentry and (width is None or height is None):
            raise ValueError(
                "frame_width and frame_height are required when border re-entry is enabled"
            )


@dataclass(frozen=True)
class SequenceGraphSummary:
    sequence: str
    seed_count: int
    input_proposals: int
    retained_proposals: int
    tracklet_count: int
    candidate_links: int
    selected_links: int
    selected_border_links: int
    path_count: int
    seeded_paths: int
    late_birth_paths: int
    dropped_paths: int
    output_rows: int


@dataclass(frozen=True)
class SelectedGraphLink:
    sequence: str
    source_tracklet: int
    target_tracklet: int
    source_end_frame: int
    target_start_frame: int
    gap_frames: int
    cost: float
    gain: float
    border_reentry: bool


@dataclass(frozen=True)
class GlobalTrackletGraphSummary:
    schema: str
    proposal_path: str
    first_frame_label_dir: str
    output_dir: str
    parameters: GlobalTrackletGraphParameters
    selected_sequences: tuple[str, ...]
    sequence_count: int
    seed_count: int
    input_proposals: int
    retained_proposals: int
    tracklet_count: int
    candidate_links: int
    selected_links: int
    selected_border_links: int
    path_count: int
    seeded_paths: int
    late_birth_paths: int
    dropped_paths: int
    output_rows: int
    sequences: tuple[SequenceGraphSummary, ...]
    links: tuple[SelectedGraphLink, ...]


@dataclass
class _Tracklet:
    index: int
    rows: list[Detection]
    contains_seed: bool

    @property
    def start_frame(self) -> int:
        return self.rows[0].frame_id

    @property
    def end_frame(self) -> int:
        return self.rows[-1].frame_id

    @property
    def length(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class _LinkCandidate:
    source: int
    target: int
    gap: int
    cost: float
    gain: float
    border_reentry: bool


@dataclass(frozen=True)
class _SequenceResult:
    rows: tuple[Detection, ...]
    summary: SequenceGraphSummary
    links: tuple[SelectedGraphLink, ...]


def track_global_proposal_graph(
    proposal_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    parameters: GlobalTrackletGraphParameters | None = None,
    sequences: Iterable[str] | None = None,
) -> GlobalTrackletGraphSummary:
    """Track detector proposals without using truth or hidden test information."""

    params = parameters or GlobalTrackletGraphParameters()
    _validate_paths(proposal_path, first_frame_label_dir, output_dir)
    label_paths = sorted(first_frame_label_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(
            f"first-frame label directory contains no .txt files: {first_frame_label_dir}"
        )
    labels_by_sequence = {path.stem: path for path in label_paths}
    requested = set(sequences or ())
    if requested:
        missing = sorted(requested - set(labels_by_sequence))
        if missing:
            raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")
    selected_sequences = tuple(
        name for name in sorted(labels_by_sequence) if not requested or name in requested
    )
    proposal_texts = prediction_texts(proposal_path)
    expected_names = {f"{name}.txt" for name in labels_by_sequence}
    unexpected = sorted(set(proposal_texts) - expected_names)
    if unexpected:
        raise ValueError(
            "proposal input contains unknown sequence files: " + ", ".join(unexpected)
        )

    results: list[_SequenceResult] = []
    for sequence in selected_sequences:
        label_path = labels_by_sequence[sequence]
        seeds = parse_detection_text(
            label_path.read_text(encoding="utf-8"),
            source=str(label_path),
        )
        _validate_seed_rows(seeds, source=label_path)
        seeds = sorted(seeds, key=lambda row: row.object_id)
        proposal_source = f"{proposal_path}:{sequence}.txt"
        proposals = parse_detection_text(
            proposal_texts.get(f"{sequence}.txt", ""),
            source=proposal_source,
        )
        _validate_proposal_confidence(proposals, source=proposal_source)
        results.append(_track_sequence(sequence, seeds, proposals, params))

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.txt"):
        stale_path.unlink()
    for result in results:
        (output_dir / f"{result.summary.sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in result.rows),
            encoding="utf-8",
        )

    summaries = tuple(result.summary for result in results)
    links = tuple(link for result in results for link in result.links)
    summary = GlobalTrackletGraphSummary(
        schema="raft-uav-multi-uav-lts-global-tracklet-graph-v1",
        proposal_path=str(proposal_path),
        first_frame_label_dir=str(first_frame_label_dir),
        output_dir=str(output_dir),
        parameters=params,
        selected_sequences=selected_sequences,
        sequence_count=len(summaries),
        seed_count=sum(row.seed_count for row in summaries),
        input_proposals=sum(row.input_proposals for row in summaries),
        retained_proposals=sum(row.retained_proposals for row in summaries),
        tracklet_count=sum(row.tracklet_count for row in summaries),
        candidate_links=sum(row.candidate_links for row in summaries),
        selected_links=sum(row.selected_links for row in summaries),
        selected_border_links=sum(row.selected_border_links for row in summaries),
        path_count=sum(row.path_count for row in summaries),
        seeded_paths=sum(row.seeded_paths for row in summaries),
        late_birth_paths=sum(row.late_birth_paths for row in summaries),
        dropped_paths=sum(row.dropped_paths for row in summaries),
        output_rows=sum(row.output_rows for row in summaries),
        sequences=summaries,
        links=links,
    )
    _write_summary_artifacts(summary, output_dir)
    return summary


def _track_sequence(
    sequence: str,
    seeds: Sequence[Detection],
    proposals: Sequence[Detection],
    params: GlobalTrackletGraphParameters,
) -> _SequenceResult:
    prepared = _prepare_proposals(proposals, seeds, params)
    seed_set = set(seeds)
    tracklets = _build_local_tracklets(prepared, seed_set=seed_set, params=params)
    candidates = _candidate_links(tracklets, params=params)
    selected = _select_sparse_links(tracklets, candidates)
    paths = _paths_from_links(tracklets, selected)
    output_rows, seeded_paths, late_birth_paths, dropped_paths = _materialize_paths(
        paths,
        seeds,
        params=params,
    )
    selected_links = tuple(
        SelectedGraphLink(
            sequence=sequence,
            source_tracklet=item.source,
            target_tracklet=item.target,
            source_end_frame=tracklets[item.source].end_frame,
            target_start_frame=tracklets[item.target].start_frame,
            gap_frames=item.gap,
            cost=item.cost,
            gain=item.gain,
            border_reentry=item.border_reentry,
        )
        for item in selected
    )
    summary = SequenceGraphSummary(
        sequence=sequence,
        seed_count=len(seeds),
        input_proposals=len(proposals),
        retained_proposals=len(prepared),
        tracklet_count=len(tracklets),
        candidate_links=len(candidates),
        selected_links=len(selected),
        selected_border_links=sum(item.border_reentry for item in selected),
        path_count=len(paths),
        seeded_paths=seeded_paths,
        late_birth_paths=late_birth_paths,
        dropped_paths=dropped_paths,
        output_rows=len(output_rows),
    )
    return _SequenceResult(output_rows, summary, selected_links)


def _prepare_proposals(
    proposals: Sequence[Detection],
    seeds: Sequence[Detection],
    params: GlobalTrackletGraphParameters,
) -> tuple[Detection, ...]:
    by_frame: dict[int, list[Detection]] = {}
    for row in proposals:
        if row.confidence >= params.min_confidence:
            by_frame.setdefault(row.frame_id, []).append(row)
    retained: list[Detection] = []
    for frame_id in sorted(by_frame):
        rows = _nms_rows(by_frame[frame_id], threshold=params.nms_iou)
        if frame_id == 1 and seeds:
            rows = [
                row
                for row in rows
                if max(box_iou(row, seed) for seed in seeds) < params.seed_min_iou
            ]
        retained.extend(rows)
    retained.extend(seeds)
    return tuple(sorted(retained, key=_row_sort_key))


def _nms_rows(rows: Sequence[Detection], *, threshold: float) -> list[Detection]:
    ordered = sorted(rows, key=_proposal_priority_key)
    kept: list[Detection] = []
    for candidate in ordered:
        if any(box_iou(candidate, current) >= threshold for current in kept):
            continue
        kept.append(candidate)
    return sorted(kept, key=_row_sort_key)


def _build_local_tracklets(
    rows: Sequence[Detection],
    *,
    seed_set: set[Detection],
    params: GlobalTrackletGraphParameters,
) -> list[_Tracklet]:
    by_frame: dict[int, list[Detection]] = {}
    for row in rows:
        by_frame.setdefault(row.frame_id, []).append(row)
    tracklets: list[_Tracklet] = []
    active: list[_Tracklet] = []
    previous_frame: int | None = None
    for frame_id in sorted(by_frame):
        frame_rows = sorted(by_frame[frame_id], key=_row_sort_key)
        if previous_frame is None or frame_id != previous_frame + 1 or not active:
            active = [
                _new_tracklet(tracklets, row, contains_seed=row in seed_set)
                for row in frame_rows
            ]
            previous_frame = frame_id
            continue
        costs = np.full((len(active), len(frame_rows)), np.inf, dtype=float)
        for left_index, tracklet in enumerate(active):
            for right_index, row in enumerate(frame_rows):
                value = _local_link_cost(tracklet, row, params=params)
                if value is not None:
                    costs[left_index, right_index] = value
        assignments = _local_assignments(costs, params=params)
        matched_rows: set[int] = set()
        next_active: list[_Tracklet] = []
        for left_index, right_index in assignments:
            tracklet = active[left_index]
            tracklet.rows.append(frame_rows[right_index])
            tracklet.contains_seed = tracklet.contains_seed or frame_rows[right_index] in seed_set
            matched_rows.add(right_index)
            next_active.append(tracklet)
        for right_index, row in enumerate(frame_rows):
            if right_index not in matched_rows:
                next_active.append(
                    _new_tracklet(tracklets, row, contains_seed=row in seed_set)
                )
        active = sorted(next_active, key=lambda item: item.index)
        previous_frame = frame_id
    return tracklets


def _new_tracklet(
    tracklets: list[_Tracklet],
    row: Detection,
    *,
    contains_seed: bool,
) -> _Tracklet:
    tracklet = _Tracklet(len(tracklets), [row], contains_seed)
    tracklets.append(tracklet)
    return tracklet


def _local_link_cost(
    tracklet: _Tracklet,
    row: Detection,
    *,
    params: GlobalTrackletGraphParameters,
) -> float | None:
    predicted = _predict_row(tracklet.rows, target_frame=row.frame_id)
    distance = _normalized_center_distance(predicted, row)
    size_change = _log_size_change(predicted, row)
    if distance > params.local_max_normalized_distance:
        return None
    if size_change > params.local_max_log_size_change:
        return None
    overlap_cost = 1.0 - box_iou(predicted, row)
    cost = 0.65 * distance + 0.20 * size_change + 0.15 * overlap_cost
    return cost if cost <= params.local_max_cost else None


def _local_assignments(
    costs: np.ndarray,
    *,
    params: GlobalTrackletGraphParameters,
) -> list[tuple[int, int]]:
    if costs.size == 0 or not np.isfinite(costs).any():
        return []
    finite = costs[np.isfinite(costs)]
    invalid = max(_INVALID_COST, float(np.max(finite)) + _INVALID_COST)
    row_indices, column_indices = linear_sum_assignment(
        np.where(np.isfinite(costs), costs, invalid)
    )
    accepted: list[tuple[int, int]] = []
    for row_index, column_index in zip(row_indices, column_indices, strict=True):
        value = costs[row_index, column_index]
        if not math.isfinite(value) or value > params.local_max_cost:
            continue
        if params.reciprocal_local_links and not _is_reciprocal_unambiguous(
            costs,
            row_index,
            column_index,
            min_margin=params.local_min_margin,
        ):
            continue
        accepted.append((int(row_index), int(column_index)))
    return accepted


def _is_reciprocal_unambiguous(
    costs: np.ndarray,
    row_index: int,
    column_index: int,
    *,
    min_margin: float,
) -> bool:
    value = costs[row_index, column_index]
    row_values = np.sort(costs[row_index][np.isfinite(costs[row_index])])
    column_values = np.sort(costs[:, column_index][np.isfinite(costs[:, column_index])])
    if row_values.size == 0 or column_values.size == 0:
        return False
    if value > row_values[0] + _EPS or value > column_values[0] + _EPS:
        return False
    row_margin = math.inf if row_values.size == 1 else float(row_values[1] - row_values[0])
    column_margin = (
        math.inf
        if column_values.size == 1
        else float(column_values[1] - column_values[0])
    )
    return row_margin + _EPS >= min_margin and column_margin + _EPS >= min_margin


def _candidate_links(
    tracklets: Sequence[_Tracklet],
    *,
    params: GlobalTrackletGraphParameters,
) -> list[_LinkCandidate]:
    by_start: dict[int, list[_Tracklet]] = {}
    for tracklet in tracklets:
        by_start.setdefault(tracklet.start_frame, []).append(tracklet)
    starts = sorted(by_start)
    result: list[_LinkCandidate] = []
    max_gap = params.border_max_gap if params.enable_border_reentry else params.global_max_gap
    for source in tracklets:
        earliest = source.end_frame + 1
        latest = source.end_frame + max_gap + 1
        left = bisect_left(starts, earliest)
        right = bisect_right(starts, latest)
        source_candidates: list[_LinkCandidate] = []
        for start_frame in starts[left:right]:
            for target in by_start[start_frame]:
                candidate = _global_link_candidate(source, target, params=params)
                if candidate is not None:
                    source_candidates.append(candidate)
        source_candidates.sort(
            key=lambda item: (
                item.cost,
                tracklets[item.target].start_frame,
                tracklets[item.target].rows[0].center_x,
                tracklets[item.target].rows[0].center_y,
                item.target,
            )
        )
        result.extend(source_candidates[: params.edge_limit_per_tracklet])
    return result


def _global_link_candidate(
    source: _Tracklet,
    target: _Tracklet,
    *,
    params: GlobalTrackletGraphParameters,
) -> _LinkCandidate | None:
    if target.start_frame <= source.end_frame:
        return None
    if (
        source.length < params.min_link_tracklet_length
        and not source.contains_seed
    ):
        return None
    if target.length < params.min_link_tracklet_length and not target.contains_seed:
        return None
    gap = target.start_frame - source.end_frame - 1
    border = False
    if gap > params.global_max_gap:
        if not params.enable_border_reentry or gap > params.border_max_gap:
            return None
        border = _shared_border(source.rows[-1], target.rows[0], params=params)
        if not border:
            return None
    size_change = _log_size_change(source.rows[-1], target.rows[0])
    if size_change > params.max_global_log_size_change:
        return None
    if border:
        distance = _border_tangential_distance(
            source.rows[-1],
            target.rows[0],
            params=params,
        )
        velocity_mismatch = 0.0
        overlap_cost = 1.0
        gap_scale = max(1, params.border_max_gap)
    else:
        predicted = _predict_row(source.rows, target_frame=target.start_frame)
        distance = _normalized_center_distance(predicted, target.rows[0])
        distance_gate = params.max_global_normalized_distance + 0.35 * gap
        if distance > distance_gate:
            return None
        velocity_mismatch = _velocity_mismatch(source.rows, target.rows)
        if velocity_mismatch > params.max_global_velocity_mismatch:
            return None
        overlap_cost = 1.0 - box_iou(predicted, target.rows[0])
        gap_scale = max(1, params.global_max_gap)
    confidence = 0.5 * (
        _proposal_confidence(source.rows[-1]) + _proposal_confidence(target.rows[0])
    )
    cost = (
        0.50 * distance
        + 0.20 * size_change
        + 0.20 * velocity_mismatch
        + 0.10 * overlap_cost
        + 0.10 * gap / gap_scale
        - 0.05 * confidence
        + (params.border_link_penalty if border else 0.0)
    )
    if cost > params.max_link_cost:
        return None
    gain = params.birth_cost + params.death_cost - cost
    if gain <= 1e-9:
        return None
    return _LinkCandidate(source.index, target.index, gap, cost, gain, border)


def _select_sparse_links(
    tracklets: Sequence[_Tracklet],
    candidates: Sequence[_LinkCandidate],
) -> list[_LinkCandidate]:
    count = len(tracklets)
    if count == 0 or not candidates:
        return []
    by_pair = {(item.source, item.target): item for item in candidates}
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    tie_scale = 1e-10 / max(1, count * count)
    for item in candidates:
        row_indices.append(item.source)
        column_indices.append(item.target)
        tie_rank = item.source * count + item.target + 1
        data.append(-item.gain + tie_scale * tie_rank)
    for index in range(count):
        row_indices.append(index)
        column_indices.append(count + index)
        data.append(1e-12 * (index + 1))
    matrix = coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(count, 2 * count),
        dtype=float,
    ).tocsr()
    matched_rows, matched_columns = min_weight_full_bipartite_matching(matrix)
    selected = [
        by_pair[(int(row), int(column))]
        for row, column in zip(matched_rows, matched_columns, strict=True)
        if int(column) < count and (int(row), int(column)) in by_pair
    ]
    return sorted(selected, key=lambda item: (item.source, item.target))


def _paths_from_links(
    tracklets: Sequence[_Tracklet],
    links: Sequence[_LinkCandidate],
) -> list[tuple[_Tracklet, ...]]:
    successor = {item.source: item.target for item in links}
    predecessor = {item.target: item.source for item in links}
    roots = [item.index for item in tracklets if item.index not in predecessor]
    roots.sort(key=lambda index: _tracklet_sort_key(tracklets[index]))
    paths: list[tuple[_Tracklet, ...]] = []
    visited: set[int] = set()
    for root in roots:
        current = root
        path: list[_Tracklet] = []
        while current not in visited:
            visited.add(current)
            path.append(tracklets[current])
            if current not in successor:
                break
            current = successor[current]
        paths.append(tuple(path))
    for tracklet in tracklets:
        if tracklet.index not in visited:
            paths.append((tracklet,))
    return sorted(paths, key=_path_sort_key)


def _materialize_paths(
    paths: Sequence[tuple[_Tracklet, ...]],
    seeds: Sequence[Detection],
    *,
    params: GlobalTrackletGraphParameters,
) -> tuple[tuple[Detection, ...], int, int, int]:
    path_rows = [_deduplicate_path(path) for path in paths]
    mapping = _seed_path_mapping(seeds, path_rows, min_iou=params.seed_min_iou)
    seed_by_id = {seed.object_id: seed for seed in seeds}
    seeded_paths = len(mapping)
    output: list[Detection] = []
    for path_index, seed_id in sorted(mapping.items(), key=lambda item: item[1]):
        output.extend(_rows_with_identity(path_rows[path_index], seed_by_id[seed_id]))
    mapped_seed_ids = set(mapping.values())
    for seed in seeds:
        if seed.object_id not in mapped_seed_ids:
            output.append(seed)
            seeded_paths += 1

    birth_candidates: list[tuple[int, tuple[Detection, ...]]] = []
    dropped = 0
    for path_index, rows in enumerate(path_rows):
        if path_index in mapping:
            continue
        if not _eligible_birth(rows, params=params):
            dropped += 1
            continue
        birth_candidates.append((path_index, rows))
    birth_candidates.sort(
        key=lambda item: (
            item[1][0].frame_id,
            item[1][0].center_x,
            item[1][0].center_y,
            -len(item[1]),
            item[0],
        )
    )
    next_id = max((seed.object_id for seed in seeds), default=0) + 1
    for _path_index, rows in birth_candidates:
        output.extend(replace(row, object_id=next_id) for row in rows)
        next_id += 1
    output_rows = tuple(sorted(output, key=lambda row: (row.frame_id, row.object_id)))
    reject_duplicate_keys(list(output_rows), label="global-graph output")
    return output_rows, seeded_paths, len(birth_candidates), dropped


def _seed_path_mapping(
    seeds: Sequence[Detection],
    paths: Sequence[tuple[Detection, ...]],
    *,
    min_iou: float,
) -> dict[int, int]:
    frame_one_rows: list[Detection] = []
    path_indices: list[int] = []
    for path_index, rows in enumerate(paths):
        frame_one = [row for row in rows if row.frame_id == 1]
        if frame_one:
            frame_one_rows.append(frame_one[0])
            path_indices.append(path_index)
    if not seeds or not frame_one_rows:
        return {}
    overlaps = iou_matrix(tuple(seeds), tuple(frame_one_rows))
    valid = overlaps >= min_iou
    cardinality_bonus = float(min(overlaps.shape) + 1)
    score = np.where(valid, cardinality_bonus + overlaps, 0.0)
    seed_indices, row_indices = linear_sum_assignment(-score)
    return {
        path_indices[row_index]: seeds[seed_index].object_id
        for seed_index, row_index in zip(seed_indices, row_indices, strict=True)
        if valid[seed_index, row_index]
    }


def _rows_with_identity(
    rows: Sequence[Detection],
    seed: Detection,
) -> list[Detection]:
    result: list[Detection] = []
    for row in rows:
        if row.frame_id == 1:
            result.append(seed)
        else:
            result.append(replace(row, object_id=seed.object_id))
    if not any(row.frame_id == 1 for row in result):
        result.append(seed)
    return result


def _eligible_birth(
    rows: Sequence[Detection],
    *,
    params: GlobalTrackletGraphParameters,
) -> bool:
    if not params.allow_births or not rows:
        return False
    if rows[0].frame_id == 1 and not params.allow_frame_one_unseeded_births:
        return False
    frame_ids = {row.frame_id for row in rows}
    if len(frame_ids) < params.min_birth_frames:
        return False
    if max(frame_ids) - min(frame_ids) < params.min_birth_span:
        return False
    mean_confidence = float(np.mean([_proposal_confidence(row) for row in rows]))
    return mean_confidence + _EPS >= params.min_birth_mean_confidence


def _deduplicate_path(path: Sequence[_Tracklet]) -> tuple[Detection, ...]:
    by_frame: dict[int, Detection] = {}
    for tracklet in path:
        for row in tracklet.rows:
            current = by_frame.get(row.frame_id)
            if current is None or _proposal_priority_key(row) < _proposal_priority_key(current):
                by_frame[row.frame_id] = row
    return tuple(by_frame[frame_id] for frame_id in sorted(by_frame))


def _predict_row(rows: Sequence[Detection], *, target_frame: int) -> Detection:
    last = rows[-1]
    delta = target_frame - last.frame_id
    if delta <= 0 or len(rows) < 2:
        return replace(last, frame_id=target_frame)
    previous = rows[-2]
    history_delta = last.frame_id - previous.frame_id
    if history_delta <= 0:
        return replace(last, frame_id=target_frame)
    velocity_x = (last.center_x - previous.center_x) / history_delta
    velocity_y = (last.center_y - previous.center_y) / history_delta
    velocity_log_width = math.log(last.width / previous.width) / history_delta
    velocity_log_height = math.log(last.height / previous.height) / history_delta
    center_x = last.center_x + velocity_x * delta
    center_y = last.center_y + velocity_y * delta
    width = last.width * math.exp(np.clip(velocity_log_width * delta, -3.0, 3.0))
    height = last.height * math.exp(np.clip(velocity_log_height * delta, -3.0, 3.0))
    return replace(
        last,
        frame_id=target_frame,
        x1=center_x - 0.5 * width,
        y1=center_y - 0.5 * height,
        width=width,
        height=height,
    )


def _velocity_mismatch(
    source_rows: Sequence[Detection],
    target_rows: Sequence[Detection],
) -> float:
    if len(source_rows) < 2 or len(target_rows) < 2:
        return 0.0
    source_velocity = _velocity(source_rows[-2], source_rows[-1])
    target_velocity = _velocity(target_rows[0], target_rows[1])
    scale = max(
        4.0,
        math.sqrt(source_rows[-1].width * source_rows[-1].height),
        math.sqrt(target_rows[0].width * target_rows[0].height),
    )
    return math.hypot(
        source_velocity[0] - target_velocity[0],
        source_velocity[1] - target_velocity[1],
    ) / scale


def _velocity(left: Detection, right: Detection) -> tuple[float, float]:
    delta = right.frame_id - left.frame_id
    if delta <= 0:
        return (0.0, 0.0)
    return (
        (right.center_x - left.center_x) / delta,
        (right.center_y - left.center_y) / delta,
    )


def _normalized_center_distance(left: Detection, right: Detection) -> float:
    scale = max(
        4.0,
        math.sqrt(left.width * left.height),
        math.sqrt(right.width * right.height),
    )
    return math.hypot(left.center_x - right.center_x, left.center_y - right.center_y) / scale


def _log_size_change(left: Detection, right: Detection) -> float:
    return abs(math.log(right.width / left.width)) + abs(math.log(right.height / left.height))


def _shared_border(
    left: Detection,
    right: Detection,
    *,
    params: GlobalTrackletGraphParameters,
) -> bool:
    left_borders = _near_borders(left, params=params)
    right_borders = _near_borders(right, params=params)
    return bool(left_borders & right_borders)


def _near_borders(
    row: Detection,
    *,
    params: GlobalTrackletGraphParameters,
) -> set[str]:
    assert params.frame_width is not None
    assert params.frame_height is not None
    margin = params.border_margin_px
    borders: set[str] = set()
    if row.x1 <= margin:
        borders.add("left")
    if row.x1 + row.width >= params.frame_width - margin:
        borders.add("right")
    if row.y1 <= margin:
        borders.add("top")
    if row.y1 + row.height >= params.frame_height - margin:
        borders.add("bottom")
    return borders


def _border_tangential_distance(
    left: Detection,
    right: Detection,
    *,
    params: GlobalTrackletGraphParameters,
) -> float:
    shared = _near_borders(left, params=params) & _near_borders(right, params=params)
    scale = max(
        4.0,
        math.sqrt(left.width * left.height),
        math.sqrt(right.width * right.height),
    )
    distances = []
    for border in shared:
        if border in {"left", "right"}:
            distances.append(abs(left.center_y - right.center_y) / scale)
        else:
            distances.append(abs(left.center_x - right.center_x) / scale)
    return min(distances, default=math.inf)


def _proposal_confidence(row: Detection) -> float:
    return float(np.clip(row.confidence, 0.0, 1.0))


def _proposal_priority_key(row: Detection) -> tuple[float, ...]:
    return (
        -_proposal_confidence(row),
        row.x1,
        row.y1,
        row.width,
        row.height,
        float(row.object_id),
    )


def _row_sort_key(row: Detection) -> tuple[float, ...]:
    return (
        float(row.frame_id),
        row.center_x,
        row.center_y,
        row.width,
        row.height,
        -_proposal_confidence(row),
        float(row.object_id),
    )


def _tracklet_sort_key(tracklet: _Tracklet) -> tuple[float, ...]:
    first = tracklet.rows[0]
    return (
        float(tracklet.start_frame),
        first.center_x,
        first.center_y,
        -float(tracklet.length),
        float(tracklet.index),
    )


def _path_sort_key(path: tuple[_Tracklet, ...]) -> tuple[float, ...]:
    return _tracklet_sort_key(path[0])


def _validate_paths(
    proposal_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
) -> None:
    if not proposal_path.exists():
        raise FileNotFoundError(f"proposal input does not exist: {proposal_path}")
    if not first_frame_label_dir.exists():
        raise FileNotFoundError(
            f"first-frame label directory does not exist: {first_frame_label_dir}"
        )
    if not first_frame_label_dir.is_dir():
        raise NotADirectoryError(
            f"first-frame label path is not a directory: {first_frame_label_dir}"
        )
    output = output_dir.resolve()
    labels = first_frame_label_dir.resolve()
    if _paths_overlap(output, labels):
        raise ValueError(
            "output directory must be disjoint from first-frame label directory"
        )
    if proposal_path.is_dir() and _paths_overlap(output, proposal_path.resolve()):
        raise ValueError("output directory must be disjoint from proposal directory")



def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents

def _validate_seed_rows(rows: Sequence[Detection], *, source: Path) -> None:
    if any(row.frame_id != 1 for row in rows):
        raise ValueError(f"{source}: expected first-frame-only labels")
    reject_duplicate_keys(list(rows), label="seed")


def _validate_proposal_confidence(rows: Sequence[Detection], *, source: str) -> None:
    for row in rows:
        if not 0.0 <= row.confidence <= 1.0:
            raise ValueError(f"{source}: proposal confidence must be in [0, 1]")


def _strict_bool(value: object, *, name: str) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    raise ValueError(f"{name} must be a Boolean")


def _optional_positive_finite(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    parsed = validate_nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive when provided")
    return parsed


def _write_summary_artifacts(
    summary: GlobalTrackletGraphSummary,
    output_dir: Path,
) -> None:
    (output_dir / "global_tracklet_graph_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sequence_fields = [field.name for field in SequenceGraphSummary.__dataclass_fields__.values()]
    with (output_dir / "global_tracklet_graph_sequences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sequence_fields)
        writer.writeheader()
        for row in summary.sequences:
            writer.writerow(asdict(row))
    link_fields = [field.name for field in SelectedGraphLink.__dataclass_fields__.values()]
    with (output_dir / "global_tracklet_graph_links.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=link_fields)
        writer.writeheader()
        for row in summary.links:
            writer.writerow(asdict(row))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_path", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--min-confidence", type=float, default=0.003)
    parser.add_argument("--nms-iou", type=float, default=0.95)
    parser.add_argument(
        "--reciprocal-local-links",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local-max-normalized-distance", type=float, default=3.0)
    parser.add_argument("--local-max-log-size-change", type=float, default=1.25)
    parser.add_argument("--local-max-cost", type=float, default=2.25)
    parser.add_argument("--local-min-margin", type=float, default=0.05)
    parser.add_argument("--global-max-gap", type=int, default=12)
    parser.add_argument("--max-global-normalized-distance", type=float, default=6.0)
    parser.add_argument("--max-global-log-size-change", type=float, default=1.75)
    parser.add_argument("--max-global-velocity-mismatch", type=float, default=4.0)
    parser.add_argument("--max-link-cost", type=float, default=4.5)
    parser.add_argument("--birth-cost", type=float, default=2.5)
    parser.add_argument("--death-cost", type=float, default=2.5)
    parser.add_argument("--edge-limit-per-tracklet", type=int, default=8)
    parser.add_argument("--min-link-tracklet-length", type=int, default=2)
    parser.add_argument(
        "--allow-births",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-frame-one-unseeded-births",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--min-birth-frames", type=int, default=3)
    parser.add_argument("--min-birth-span", type=int, default=2)
    parser.add_argument("--min-birth-mean-confidence", type=float, default=0.01)
    parser.add_argument("--seed-min-iou", type=float, default=0.25)
    parser.add_argument(
        "--enable-border-reentry",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--border-max-gap", type=int, default=90)
    parser.add_argument("--border-margin-px", type=float, default=32.0)
    parser.add_argument("--border-link-penalty", type=float, default=0.75)
    parser.add_argument("--frame-width", type=float)
    parser.add_argument("--frame-height", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    parameters = GlobalTrackletGraphParameters(
        min_confidence=args.min_confidence,
        nms_iou=args.nms_iou,
        reciprocal_local_links=args.reciprocal_local_links,
        local_max_normalized_distance=args.local_max_normalized_distance,
        local_max_log_size_change=args.local_max_log_size_change,
        local_max_cost=args.local_max_cost,
        local_min_margin=args.local_min_margin,
        global_max_gap=args.global_max_gap,
        max_global_normalized_distance=args.max_global_normalized_distance,
        max_global_log_size_change=args.max_global_log_size_change,
        max_global_velocity_mismatch=args.max_global_velocity_mismatch,
        max_link_cost=args.max_link_cost,
        birth_cost=args.birth_cost,
        death_cost=args.death_cost,
        edge_limit_per_tracklet=args.edge_limit_per_tracklet,
        min_link_tracklet_length=args.min_link_tracklet_length,
        allow_births=args.allow_births,
        allow_frame_one_unseeded_births=args.allow_frame_one_unseeded_births,
        min_birth_frames=args.min_birth_frames,
        min_birth_span=args.min_birth_span,
        min_birth_mean_confidence=args.min_birth_mean_confidence,
        seed_min_iou=args.seed_min_iou,
        enable_border_reentry=args.enable_border_reentry,
        border_max_gap=args.border_max_gap,
        border_margin_px=args.border_margin_px,
        border_link_penalty=args.border_link_penalty,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    summary = track_global_proposal_graph(
        args.proposal_path,
        args.first_frame_label_dir,
        args.output_dir,
        parameters=parameters,
        sequences=args.sequences,
    )
    print(f"sequence_count={summary.sequence_count}")
    print(f"seeded_paths={summary.seeded_paths}")
    print(f"late_birth_paths={summary.late_birth_paths}")
    print(f"selected_links={summary.selected_links}")
    print(f"output_rows={summary.output_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
