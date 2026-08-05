"""Track Multi-UAV LTS proposal banks with a fixed first-frame identity set."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import (
    Detection,
    box_iou,
    format_detection,
    parse_detection_text,
    reject_duplicate_keys,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)
from .metrics import evaluate_lts_predictions
from .proposal_oracle import (
    _ProposalReader,
    _canonicalize_proposals,
    _normalize_proposal_paths,
)

_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_LARGE_COST = 1e12
_EPS = 1e-12


@dataclass(frozen=True)
class FixedLabelTrackerConfig:
    min_confidence: float
    duplicate_iou_threshold: float
    max_candidates_per_frame: int
    center_weight: float
    iou_weight: float
    scale_weight: float
    confidence_weight: float
    max_center_distance: float
    center_gate_growth: float
    max_log_scale_change: float
    scale_gate_growth: float
    max_assignment_cost: float
    missed_cost: float
    missed_cost_growth: float
    max_missed_frames: int
    coast_frames: int
    coast_confidence_decay: float
    coast_conflict_iou: float
    velocity_smoothing: float
    velocity_decay: float
    global_motion_smoothing: float


@dataclass(frozen=True)
class TrackerProposalSource:
    name: str
    path: str


@dataclass(frozen=True)
class TrackerScore:
    codabench_hota: float
    codabench_mota: float
    codabench_idf1: float
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    idf1: float


@dataclass(frozen=True)
class SequenceTrackerSummary:
    sequence: str
    frame_count: int
    seed_count: int
    input_proposals: int
    candidate_proposals: int
    assigned_rows: int
    coasted_rows: int
    suppressed_coasts: int
    missed_states: int
    unassigned_candidates: int
    mean_assignment_cost: float
    global_motion_updates: int
    assignments_by_source: Mapping[str, int]


@dataclass(frozen=True)
class FixedLabelTrackerSummary:
    schema: str
    first_frame_label_dir: str
    sequence_root: str | None
    truth_dir: str | None
    output_dir: str
    selected_sequences: tuple[str, ...]
    proposal_sources: tuple[TrackerProposalSource, ...]
    config: FixedLabelTrackerConfig
    sequence_count: int
    seed_count: int
    input_proposals: int
    candidate_proposals: int
    assigned_rows: int
    coasted_rows: int
    suppressed_coasts: int
    missed_states: int
    unassigned_candidates: int
    assignments_by_source: Mapping[str, int]
    score: TrackerScore | None
    sequences: tuple[SequenceTrackerSummary, ...]


@dataclass(frozen=True)
class _Candidate:
    row: Detection
    source: str


@dataclass
class _TrackState:
    object_id: int
    box: Detection
    residual_velocity_x: float = 0.0
    residual_velocity_y: float = 0.0
    log_width_velocity: float = 0.0
    log_height_velocity: float = 0.0
    missed: int = 0


@dataclass(frozen=True)
class _Assignment:
    state_index: int
    candidate_index: int | None
    cost: float


def track_fixed_label_proposals(
    proposal_paths: Mapping[str, Path],
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    sequence_root: Path | None = None,
    truth_dir: Path | None = None,
    min_confidence: float = 0.003,
    duplicate_iou_threshold: float = 0.98,
    max_candidates_per_frame: int = 500,
    center_weight: float = 1.0,
    iou_weight: float = 0.25,
    scale_weight: float = 0.25,
    confidence_weight: float = 0.1,
    max_center_distance: float = 6.0,
    center_gate_growth: float = 0.35,
    max_log_scale_change: float = 1.5,
    scale_gate_growth: float = 0.15,
    max_assignment_cost: float = 8.0,
    missed_cost: float = 2.5,
    missed_cost_growth: float = 0.1,
    max_missed_frames: int = 60,
    coast_frames: int = 0,
    coast_confidence_decay: float = 0.5,
    coast_conflict_iou: float = 0.5,
    velocity_smoothing: float = 0.5,
    velocity_decay: float = 0.9,
    global_motion_smoothing: float = 0.5,
    sequences: Iterable[str] | None = None,
) -> FixedLabelTrackerSummary:
    """Track a closed identity population over one or more proposal sources."""

    if not proposal_paths:
        raise ValueError("at least one proposal source is required")
    normalized_paths = _normalize_proposal_paths(proposal_paths)
    config = _validated_config(
        min_confidence=min_confidence,
        duplicate_iou_threshold=duplicate_iou_threshold,
        max_candidates_per_frame=max_candidates_per_frame,
        center_weight=center_weight,
        iou_weight=iou_weight,
        scale_weight=scale_weight,
        confidence_weight=confidence_weight,
        max_center_distance=max_center_distance,
        center_gate_growth=center_gate_growth,
        max_log_scale_change=max_log_scale_change,
        scale_gate_growth=scale_gate_growth,
        max_assignment_cost=max_assignment_cost,
        missed_cost=missed_cost,
        missed_cost_growth=missed_cost_growth,
        max_missed_frames=max_missed_frames,
        coast_frames=coast_frames,
        coast_confidence_decay=coast_confidence_decay,
        coast_conflict_iou=coast_conflict_iou,
        velocity_smoothing=velocity_smoothing,
        velocity_decay=velocity_decay,
        global_motion_smoothing=global_motion_smoothing,
    )
    label_paths = _first_frame_label_paths(first_frame_label_dir)
    label_names = {path.stem for path in label_paths}
    selected_names = _selected_sequences(label_names, sequences)
    selected_paths = [path for path in label_paths if path.stem in selected_names]
    _validate_sequence_root(sequence_root, selected_names)
    _validate_truth_dir(truth_dir, selected_names)
    _reject_output_aliases(
        normalized_paths,
        first_frame_label_dir,
        sequence_root,
        truth_dir,
        output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.txt"):
        stale_path.unlink()

    summaries: list[SequenceTrackerSummary] = []
    aggregate_sources: Counter[str] = Counter()
    with ExitStack() as stack:
        readers: dict[str, _ProposalReader] = {}
        for name, path in normalized_paths.items():
            reader = _ProposalReader(path, truth_names=label_names)
            stack.callback(reader.close)
            readers[name] = reader
        for label_path in selected_paths:
            sequence = label_path.stem
            seeds = parse_detection_text(
                label_path.read_text(encoding="utf-8"),
                source=str(label_path),
            )
            _validate_seed_rows(seeds, label_path)
            rows_by_source = {
                name: tuple(_canonicalize_proposals(reader.rows(sequence)))
                for name, reader in readers.items()
            }
            frame_count = _sequence_frame_count(
                sequence,
                rows_by_source,
                sequence_root=sequence_root,
            )
            rows, summary = _track_sequence(
                sequence,
                seeds,
                rows_by_source,
                frame_count=frame_count,
                config=config,
            )
            (output_dir / f"{sequence}.txt").write_text(
                "".join(format_detection(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summaries.append(summary)
            aggregate_sources.update(summary.assignments_by_source)

    score = _score_predictions(output_dir, truth_dir, tuple(sorted(selected_names)))
    summary = FixedLabelTrackerSummary(
        schema="raft-uav-multi-uav-lts-fixed-label-proposal-tracker-v1",
        first_frame_label_dir=str(first_frame_label_dir),
        sequence_root=None if sequence_root is None else str(sequence_root),
        truth_dir=None if truth_dir is None else str(truth_dir),
        output_dir=str(output_dir),
        selected_sequences=tuple(sorted(selected_names)),
        proposal_sources=tuple(
            TrackerProposalSource(name=name, path=str(path))
            for name, path in normalized_paths.items()
        ),
        config=config,
        sequence_count=len(summaries),
        seed_count=sum(row.seed_count for row in summaries),
        input_proposals=sum(row.input_proposals for row in summaries),
        candidate_proposals=sum(row.candidate_proposals for row in summaries),
        assigned_rows=sum(row.assigned_rows for row in summaries),
        coasted_rows=sum(row.coasted_rows for row in summaries),
        suppressed_coasts=sum(row.suppressed_coasts for row in summaries),
        missed_states=sum(row.missed_states for row in summaries),
        unassigned_candidates=sum(row.unassigned_candidates for row in summaries),
        assignments_by_source=dict(sorted(aggregate_sources.items())),
        score=score,
        sequences=tuple(summaries),
    )
    _write_summary(summary, output_dir)
    return summary


def _validated_config(**values: object) -> FixedLabelTrackerConfig:
    min_confidence = validate_unit_interval(
        values["min_confidence"], name="min_confidence"
    )
    duplicate_iou_threshold = validate_unit_interval(
        values["duplicate_iou_threshold"], name="duplicate_iou_threshold"
    )
    if duplicate_iou_threshold <= 0.0:
        raise ValueError("duplicate_iou_threshold must be positive")
    max_candidates = validate_nonnegative_int(
        values["max_candidates_per_frame"], name="max_candidates_per_frame"
    )
    if max_candidates <= 0:
        raise ValueError("max_candidates_per_frame must be positive")
    nonnegative_names = (
        "center_weight",
        "iou_weight",
        "scale_weight",
        "confidence_weight",
        "max_center_distance",
        "center_gate_growth",
        "max_log_scale_change",
        "scale_gate_growth",
        "max_assignment_cost",
        "missed_cost",
        "missed_cost_growth",
    )
    normalized = {
        name: validate_nonnegative_finite(values[name], name=name)
        for name in nonnegative_names
    }
    if not any(
        normalized[name] > 0.0
        for name in ("center_weight", "iou_weight", "scale_weight", "confidence_weight")
    ):
        raise ValueError("at least one assignment cost weight must be positive")
    if normalized["max_center_distance"] <= 0.0:
        raise ValueError("max_center_distance must be positive")
    if normalized["max_log_scale_change"] <= 0.0:
        raise ValueError("max_log_scale_change must be positive")
    if normalized["max_assignment_cost"] <= 0.0:
        raise ValueError("max_assignment_cost must be positive")
    max_missed = validate_nonnegative_int(
        values["max_missed_frames"], name="max_missed_frames"
    )
    coast_frames = validate_nonnegative_int(values["coast_frames"], name="coast_frames")
    if coast_frames > max_missed:
        raise ValueError("coast_frames must not exceed max_missed_frames")
    unit_names = (
        "coast_confidence_decay",
        "coast_conflict_iou",
        "velocity_smoothing",
        "velocity_decay",
        "global_motion_smoothing",
    )
    units = {name: validate_unit_interval(values[name], name=name) for name in unit_names}
    if units["coast_conflict_iou"] <= 0.0:
        raise ValueError("coast_conflict_iou must be positive")
    return FixedLabelTrackerConfig(
        min_confidence=min_confidence,
        duplicate_iou_threshold=duplicate_iou_threshold,
        max_candidates_per_frame=max_candidates,
        center_weight=normalized["center_weight"],
        iou_weight=normalized["iou_weight"],
        scale_weight=normalized["scale_weight"],
        confidence_weight=normalized["confidence_weight"],
        max_center_distance=normalized["max_center_distance"],
        center_gate_growth=normalized["center_gate_growth"],
        max_log_scale_change=normalized["max_log_scale_change"],
        scale_gate_growth=normalized["scale_gate_growth"],
        max_assignment_cost=normalized["max_assignment_cost"],
        missed_cost=normalized["missed_cost"],
        missed_cost_growth=normalized["missed_cost_growth"],
        max_missed_frames=max_missed,
        coast_frames=coast_frames,
        coast_confidence_decay=units["coast_confidence_decay"],
        coast_conflict_iou=units["coast_conflict_iou"],
        velocity_smoothing=units["velocity_smoothing"],
        velocity_decay=units["velocity_decay"],
        global_motion_smoothing=units["global_motion_smoothing"],
    )


def _first_frame_label_paths(first_frame_label_dir: Path) -> list[Path]:
    if not first_frame_label_dir.exists():
        raise FileNotFoundError(
            f"first-frame label directory does not exist: {first_frame_label_dir}"
        )
    if not first_frame_label_dir.is_dir():
        raise NotADirectoryError(
            f"first-frame label path is not a directory: {first_frame_label_dir}"
        )
    paths = sorted(first_frame_label_dir.glob("*.txt"))
    if not paths:
        raise ValueError(
            f"first-frame label directory contains no .txt files: {first_frame_label_dir}"
        )
    return paths


def _selected_sequences(
    available: set[str],
    sequences: Iterable[str] | None,
) -> set[str]:
    requested = set(sequences or ())
    if not requested:
        return set(available)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")
    return requested


def _validate_sequence_root(sequence_root: Path | None, sequences: set[str]) -> None:
    if sequence_root is None:
        return
    if not sequence_root.exists():
        raise FileNotFoundError(f"sequence root does not exist: {sequence_root}")
    if not sequence_root.is_dir():
        raise NotADirectoryError(f"sequence root is not a directory: {sequence_root}")
    missing = sorted(name for name in sequences if not (sequence_root / name).is_dir())
    if missing:
        raise ValueError(f"sequence root is missing sequences: {', '.join(missing)}")


def _validate_truth_dir(truth_dir: Path | None, sequences: set[str]) -> None:
    if truth_dir is None:
        return
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth directory does not exist: {truth_dir}")
    if not truth_dir.is_dir():
        raise NotADirectoryError(f"truth path is not a directory: {truth_dir}")
    missing = sorted(name for name in sequences if not (truth_dir / f"{name}.txt").is_file())
    if missing:
        raise ValueError(f"truth directory is missing sequences: {', '.join(missing)}")


def _reject_output_aliases(
    proposal_paths: Mapping[str, Path],
    first_frame_label_dir: Path,
    sequence_root: Path | None,
    truth_dir: Path | None,
    output_dir: Path,
) -> None:
    output = output_dir.resolve()
    protected = {
        "first-frame label directory": first_frame_label_dir.resolve(),
    }
    if sequence_root is not None:
        protected["sequence root"] = sequence_root.resolve()
    if truth_dir is not None:
        protected["truth directory"] = truth_dir.resolve()
    for label, path in protected.items():
        if output == path:
            raise ValueError(f"output directory must differ from {label}")
    seen_inputs: dict[Path, str] = {}
    for name, path in proposal_paths.items():
        resolved = path.resolve()
        if resolved in protected.values():
            raise ValueError(f"proposal source '{name}' aliases a protected input")
        if resolved in seen_inputs:
            raise ValueError(
                f"proposal sources '{seen_inputs[resolved]}' and '{name}' alias the same input"
            )
        seen_inputs[resolved] = name
        if path.is_dir() and output == resolved:
            raise ValueError(f"output directory must differ from proposal source '{name}'")


def _validate_seed_rows(rows: list[Detection], label_path: Path) -> None:
    reject_duplicate_keys(rows, label="seed")
    if any(row.frame_id != 1 for row in rows):
        raise ValueError(f"{label_path}: expected first-frame-only labels")


def _sequence_frame_count(
    sequence: str,
    rows_by_source: Mapping[str, Sequence[Detection]],
    *,
    sequence_root: Path | None,
) -> int:
    if sequence_root is not None:
        sequence_dir = sequence_root / sequence
        count = sum(
            path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            for path in sequence_dir.iterdir()
        )
        if count <= 0:
            raise ValueError(f"sequence contains no image files: {sequence_dir}")
        return count
    return max(
        1,
        *(
            row.frame_id
            for source_rows in rows_by_source.values()
            for row in source_rows
        ),
    )


def _track_sequence(
    sequence: str,
    seeds: list[Detection],
    rows_by_source: Mapping[str, Sequence[Detection]],
    *,
    frame_count: int,
    config: FixedLabelTrackerConfig,
) -> tuple[tuple[Detection, ...], SequenceTrackerSummary]:
    states = [
        _TrackState(object_id=row.object_id, box=row)
        for row in sorted(seeds, key=lambda row: row.object_id)
    ]
    proposals_by_frame = _proposals_by_frame(rows_by_source)
    output_rows = list(sorted(seeds, key=lambda row: row.object_id))
    source_counts: Counter[str] = Counter()
    input_proposals = sum(len(rows) for rows in rows_by_source.values())
    candidate_proposals = 0
    assigned_rows = 0
    coasted_rows = 0
    suppressed_coasts = 0
    missed_states = 0
    unassigned_candidates = 0
    assignment_costs: list[float] = []
    global_dx = 0.0
    global_dy = 0.0
    global_updates = 0

    for frame_id in range(2, frame_count + 1):
        candidates = _frame_candidates(
            proposals_by_frame.get(frame_id, ()),
            config=config,
        )
        candidate_proposals += len(candidates)
        previous_boxes = [state.box for state in states]
        missed_before = [state.missed for state in states]
        predictions = [
            _predict_box(state, frame_id=frame_id, global_dx=global_dx, global_dy=global_dy)
            for state in states
        ]
        assignments = _assign_candidates(states, predictions, candidates, config=config)
        matched = [row for row in assignments if row.candidate_index is not None]
        continuous_displacements = [
            (
                candidates[row.candidate_index].row.center_x
                - previous_boxes[row.state_index].center_x,
                candidates[row.candidate_index].row.center_y
                - previous_boxes[row.state_index].center_y,
            )
            for row in matched
            if missed_before[row.state_index] == 0 and row.candidate_index is not None
        ]
        if continuous_displacements:
            observed_dx = float(np.median([value[0] for value in continuous_displacements]))
            observed_dy = float(np.median([value[1] for value in continuous_displacements]))
            smoothing = config.global_motion_smoothing
            global_dx = (1.0 - smoothing) * global_dx + smoothing * observed_dx
            global_dy = (1.0 - smoothing) * global_dy + smoothing * observed_dy
            global_updates += 1
        else:
            global_dx *= config.velocity_decay
            global_dy *= config.velocity_decay

        assigned_frame_rows: list[Detection] = []
        assigned_candidate_indices: set[int] = set()
        missed_state_indices: list[int] = []
        for assignment in assignments:
            state = states[assignment.state_index]
            prediction = predictions[assignment.state_index]
            if assignment.candidate_index is None:
                _update_missed_state(state, prediction, config=config)
                missed_states += 1
                missed_state_indices.append(assignment.state_index)
                continue
            candidate = candidates[assignment.candidate_index]
            _update_matched_state(
                state,
                candidate.row,
                prediction=prediction,
                previous=previous_boxes[assignment.state_index],
                missed_before=missed_before[assignment.state_index],
                global_dx=global_dx,
                global_dy=global_dy,
                config=config,
            )
            output = replace(candidate.row, object_id=state.object_id)
            state.box = output
            assigned_frame_rows.append(output)
            assigned_candidate_indices.add(assignment.candidate_index)
            source_counts[candidate.source] += 1
            assigned_rows += 1
            assignment_costs.append(assignment.cost)
        unassigned_candidates += len(candidates) - len(assigned_candidate_indices)

        coast_rows: list[Detection] = []
        for state_index in missed_state_indices:
            state = states[state_index]
            if state.missed > config.coast_frames:
                continue
            coast = _coasted_row(state, config=config)
            conflicts = (*assigned_frame_rows, *coast_rows)
            if any(box_iou(coast, existing) >= config.coast_conflict_iou for existing in conflicts):
                suppressed_coasts += 1
                continue
            coast_rows.append(coast)
            coasted_rows += 1
        output_rows.extend(assigned_frame_rows)
        output_rows.extend(coast_rows)

    output_rows.sort(key=lambda row: (row.frame_id, row.object_id))
    summary = SequenceTrackerSummary(
        sequence=sequence,
        frame_count=frame_count,
        seed_count=len(seeds),
        input_proposals=input_proposals,
        candidate_proposals=candidate_proposals,
        assigned_rows=assigned_rows,
        coasted_rows=coasted_rows,
        suppressed_coasts=suppressed_coasts,
        missed_states=missed_states,
        unassigned_candidates=unassigned_candidates,
        mean_assignment_cost=(
            float(np.mean(assignment_costs)) if assignment_costs else 0.0
        ),
        global_motion_updates=global_updates,
        assignments_by_source=dict(sorted(source_counts.items())),
    )
    return tuple(output_rows), summary


def _proposals_by_frame(
    rows_by_source: Mapping[str, Sequence[Detection]],
) -> dict[int, tuple[_Candidate, ...]]:
    grouped: dict[int, list[_Candidate]] = {}
    for source, rows in rows_by_source.items():
        for row in rows:
            if row.frame_id <= 1:
                continue
            grouped.setdefault(row.frame_id, []).append(_Candidate(row=row, source=source))
    return {
        frame_id: tuple(frame_rows)
        for frame_id, frame_rows in grouped.items()
    }


def _frame_candidates(
    candidates: Sequence[_Candidate],
    *,
    config: FixedLabelTrackerConfig,
) -> tuple[_Candidate, ...]:
    ordered = sorted(
        (row for row in candidates if row.row.confidence >= config.min_confidence),
        key=lambda item: (
            -item.row.confidence,
            item.source,
            item.row.x1,
            item.row.y1,
            item.row.width,
            item.row.height,
            item.row.object_id,
        ),
    )
    deduplicated: list[_Candidate] = []
    for candidate in ordered:
        if any(
            box_iou(candidate.row, existing.row) >= config.duplicate_iou_threshold
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
        if len(deduplicated) >= config.max_candidates_per_frame:
            break
    return tuple(deduplicated)


def _predict_box(
    state: _TrackState,
    *,
    frame_id: int,
    global_dx: float,
    global_dy: float,
) -> Detection:
    center_x = state.box.center_x + global_dx + state.residual_velocity_x
    center_y = state.box.center_y + global_dy + state.residual_velocity_y
    width = max(_EPS, state.box.width * math.exp(_clamp(state.log_width_velocity, -1.0, 1.0)))
    height = max(
        _EPS,
        state.box.height * math.exp(_clamp(state.log_height_velocity, -1.0, 1.0)),
    )
    return replace(
        state.box,
        frame_id=frame_id,
        x1=center_x - 0.5 * width,
        y1=center_y - 0.5 * height,
        width=width,
        height=height,
    )


def _assign_candidates(
    states: Sequence[_TrackState],
    predictions: Sequence[Detection],
    candidates: Sequence[_Candidate],
    *,
    config: FixedLabelTrackerConfig,
) -> tuple[_Assignment, ...]:
    if not states:
        return ()
    candidate_count = len(candidates)
    matrix = np.full(
        (len(states), candidate_count + len(states)),
        _LARGE_COST,
        dtype=float,
    )
    for state_index, (state, prediction) in enumerate(zip(states, predictions, strict=True)):
        if state.missed <= config.max_missed_frames:
            for candidate_index, candidate in enumerate(candidates):
                cost = _candidate_cost(state, prediction, candidate.row, config=config)
                if math.isfinite(cost):
                    matrix[state_index, candidate_index] = cost
        matrix[state_index, candidate_count + state_index] = (
            config.missed_cost + config.missed_cost_growth * state.missed
        )
    state_indices, columns = linear_sum_assignment(matrix)
    result: list[_Assignment] = []
    for state_index, column in zip(state_indices, columns, strict=True):
        cost = float(matrix[state_index, column])
        if column < candidate_count and cost < _LARGE_COST:
            result.append(_Assignment(int(state_index), int(column), cost))
        else:
            result.append(_Assignment(int(state_index), None, cost))
    result.sort(key=lambda row: row.state_index)
    return tuple(result)


def _candidate_cost(
    state: _TrackState,
    prediction: Detection,
    candidate: Detection,
    *,
    config: FixedLabelTrackerConfig,
) -> float:
    scale = max(
        1.0,
        math.sqrt(prediction.width * prediction.height),
        math.sqrt(candidate.width * candidate.height),
    )
    center_distance = math.hypot(
        prediction.center_x - candidate.center_x,
        prediction.center_y - candidate.center_y,
    ) / scale
    center_gate = config.max_center_distance * (
        1.0 + config.center_gate_growth * min(state.missed, config.max_missed_frames)
    )
    if center_distance > center_gate:
        return math.inf
    scale_change = abs(math.log(candidate.width / prediction.width)) + abs(
        math.log(candidate.height / prediction.height)
    )
    scale_gate = config.max_log_scale_change * (
        1.0 + config.scale_gate_growth * min(state.missed, config.max_missed_frames)
    )
    if scale_change > scale_gate:
        return math.inf
    confidence_cost = -math.log(max(candidate.confidence, _EPS))
    cost = (
        config.center_weight * center_distance
        + config.iou_weight * (1.0 - box_iou(prediction, candidate))
        + config.scale_weight * scale_change
        + config.confidence_weight * confidence_cost
    )
    return cost if cost <= config.max_assignment_cost else math.inf


def _update_matched_state(
    state: _TrackState,
    candidate: Detection,
    *,
    prediction: Detection,
    previous: Detection,
    missed_before: int,
    global_dx: float,
    global_dy: float,
    config: FixedLabelTrackerConfig,
) -> None:
    smoothing = config.velocity_smoothing
    if missed_before == 0:
        target_velocity_x = candidate.center_x - previous.center_x - global_dx
        target_velocity_y = candidate.center_y - previous.center_y - global_dy
        target_log_width = math.log(candidate.width / previous.width)
        target_log_height = math.log(candidate.height / previous.height)
        state.residual_velocity_x = (
            (1.0 - smoothing) * state.residual_velocity_x
            + smoothing * target_velocity_x
        )
        state.residual_velocity_y = (
            (1.0 - smoothing) * state.residual_velocity_y
            + smoothing * target_velocity_y
        )
        state.log_width_velocity = (
            (1.0 - smoothing) * state.log_width_velocity
            + smoothing * target_log_width
        )
        state.log_height_velocity = (
            (1.0 - smoothing) * state.log_height_velocity
            + smoothing * target_log_height
        )
    else:
        divisor = float(missed_before + 1)
        state.residual_velocity_x += (
            smoothing * (candidate.center_x - prediction.center_x) / divisor
        )
        state.residual_velocity_y += (
            smoothing * (candidate.center_y - prediction.center_y) / divisor
        )
        state.log_width_velocity += (
            smoothing * math.log(candidate.width / prediction.width) / divisor
        )
        state.log_height_velocity += (
            smoothing * math.log(candidate.height / prediction.height) / divisor
        )
    state.missed = 0


def _update_missed_state(
    state: _TrackState,
    prediction: Detection,
    *,
    config: FixedLabelTrackerConfig,
) -> None:
    state.box = replace(prediction, object_id=state.object_id)
    state.residual_velocity_x *= config.velocity_decay
    state.residual_velocity_y *= config.velocity_decay
    state.log_width_velocity *= config.velocity_decay
    state.log_height_velocity *= config.velocity_decay
    state.missed += 1


def _coasted_row(
    state: _TrackState,
    *,
    config: FixedLabelTrackerConfig,
) -> Detection:
    return replace(
        state.box,
        confidence=state.box.confidence * config.coast_confidence_decay**state.missed,
        visibility=state.box.visibility * config.coast_confidence_decay**state.missed,
    )


def _score_predictions(
    output_dir: Path,
    truth_dir: Path | None,
    sequences: tuple[str, ...],
) -> TrackerScore | None:
    if truth_dir is None:
        return None
    metrics = evaluate_lts_predictions(output_dir, truth_dir, sequences=sequences)
    return TrackerScore(
        codabench_hota=metrics.codabench_hota,
        codabench_mota=metrics.codabench_mota,
        codabench_idf1=metrics.codabench_idf1,
        hota=metrics.hota,
        deta=metrics.deta,
        assa=metrics.assa,
        loca=metrics.loca,
        mota=metrics.mota,
        idf1=metrics.idf1,
    )


def _write_summary(summary: FixedLabelTrackerSummary, output_dir: Path) -> None:
    (output_dir / "fixed_label_tracker_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "fixed_label_tracker_sequences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "sequence",
            "frame_count",
            "seed_count",
            "input_proposals",
            "candidate_proposals",
            "assigned_rows",
            "coasted_rows",
            "suppressed_coasts",
            "missed_states",
            "unassigned_candidates",
            "mean_assignment_cost",
            "global_motion_updates",
            "assignments_by_source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.sequences:
            payload = asdict(row)
            payload["assignments_by_source"] = json.dumps(
                row.assignments_by_source, sort_keys=True
            )
            writer.writerow(payload)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _proposal_spec(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("proposal sources must use NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", action="append", required=True, type=_proposal_spec)
    parser.add_argument("--first-frame-label-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--truth-dir", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--min-confidence", type=float, default=0.003)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=0.98)
    parser.add_argument("--max-candidates-per-frame", type=int, default=500)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.25)
    parser.add_argument("--scale-weight", type=float, default=0.25)
    parser.add_argument("--confidence-weight", type=float, default=0.1)
    parser.add_argument("--max-center-distance", type=float, default=6.0)
    parser.add_argument("--center-gate-growth", type=float, default=0.35)
    parser.add_argument("--max-log-scale-change", type=float, default=1.5)
    parser.add_argument("--scale-gate-growth", type=float, default=0.15)
    parser.add_argument("--max-assignment-cost", type=float, default=8.0)
    parser.add_argument("--missed-cost", type=float, default=2.5)
    parser.add_argument("--missed-cost-growth", type=float, default=0.1)
    parser.add_argument("--max-missed-frames", type=int, default=60)
    parser.add_argument("--coast-frames", type=int, default=0)
    parser.add_argument("--coast-confidence-decay", type=float, default=0.5)
    parser.add_argument("--coast-conflict-iou", type=float, default=0.5)
    parser.add_argument("--velocity-smoothing", type=float, default=0.5)
    parser.add_argument("--velocity-decay", type=float, default=0.9)
    parser.add_argument("--global-motion-smoothing", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    proposal_paths = dict(args.proposal)
    if len(proposal_paths) != len(args.proposal):
        raise ValueError("proposal source names must be unique")
    summary = track_fixed_label_proposals(
        proposal_paths,
        args.first_frame_label_dir,
        args.output_dir,
        sequence_root=args.sequence_root,
        truth_dir=args.truth_dir,
        min_confidence=args.min_confidence,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        max_candidates_per_frame=args.max_candidates_per_frame,
        center_weight=args.center_weight,
        iou_weight=args.iou_weight,
        scale_weight=args.scale_weight,
        confidence_weight=args.confidence_weight,
        max_center_distance=args.max_center_distance,
        center_gate_growth=args.center_gate_growth,
        max_log_scale_change=args.max_log_scale_change,
        scale_gate_growth=args.scale_gate_growth,
        max_assignment_cost=args.max_assignment_cost,
        missed_cost=args.missed_cost,
        missed_cost_growth=args.missed_cost_growth,
        max_missed_frames=args.max_missed_frames,
        coast_frames=args.coast_frames,
        coast_confidence_decay=args.coast_confidence_decay,
        coast_conflict_iou=args.coast_conflict_iou,
        velocity_smoothing=args.velocity_smoothing,
        velocity_decay=args.velocity_decay,
        global_motion_smoothing=args.global_motion_smoothing,
        sequences=args.sequences,
    )
    print(f"fixed_label_tracker_output_dir={args.output_dir}")
    print(f"fixed_label_tracker_assigned_rows={summary.assigned_rows}")
    print(f"fixed_label_tracker_coasted_rows={summary.coasted_rows}")
    if summary.score is not None:
        print(f"CODABENCH_HOTA={summary.score.codabench_hota:.12g}")
        print(f"CODABENCH_MOTA={summary.score.codabench_mota:.12g}")
        print(f"CODABENCH_IDF1={summary.score.codabench_idf1:.12g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
