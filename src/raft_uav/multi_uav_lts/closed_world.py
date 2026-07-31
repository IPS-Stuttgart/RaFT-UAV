"""Closed-world first-frame-seeded reassociation for Multi-UAV LTS."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import (
    Detection,
    box_iou,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
    rows_by_frame,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)

_EPS = 1e-12
_INVALID_COST = 1e9


@dataclass(frozen=True)
class ClosedWorldParameters:
    """Validated controls for closed-world reassociation."""

    max_gap: int
    max_cost: float
    max_normalized_distance: float
    min_iou: float
    min_nwd: float
    center_weight: float
    nwd_weight: float
    iou_weight: float
    size_weight: float
    confidence_weight: float
    gap_weight: float
    source_continuity_bonus: float
    tiny_scale: float
    min_candidate_confidence: float
    emit_coasts: bool
    coast_max_gap: int
    coast_confidence_decay: float
    min_coast_confidence: float
    max_coast_uncertainty: float
    uncertainty_growth: float
    coast_conflict_iou: float


@dataclass(frozen=True)
class SequenceClosedWorldSummary:
    sequence: str
    seed_count: int
    input_rows: int
    output_rows: int
    max_frame: int
    frame_one_matches: int
    matched_candidate_rows: int
    dropped_candidate_rows: int
    absorbed_source_switches: int
    coasted_rows: int
    unmatched_track_frames: int


@dataclass(frozen=True)
class ClosedWorldSummary:
    prediction_path: str
    first_frame_label_dir: str
    output_dir: str
    parameters: ClosedWorldParameters
    sequence_count: int
    seed_count: int
    input_rows: int
    output_rows: int
    frame_one_matches: int
    matched_candidate_rows: int
    dropped_candidate_rows: int
    absorbed_source_switches: int
    coasted_rows: int
    unmatched_track_frames: int
    sequences: tuple[SequenceClosedWorldSummary, ...]


@dataclass
class _TrackState:
    seed_id: int
    observations: list[Detection]
    last_source_id: int | None = None

    @property
    def last_observation(self) -> Detection:
        return self.observations[-1]


@dataclass(frozen=True)
class _SequenceResult:
    rows: tuple[Detection, ...]
    summary: SequenceClosedWorldSummary


def postprocess_closed_world(
    prediction_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    max_gap: int = 15,
    max_cost: float = 2.0,
    max_normalized_distance: float = 4.0,
    min_iou: float = 0.0,
    min_nwd: float = 0.0,
    center_weight: float = 0.35,
    nwd_weight: float = 1.0,
    iou_weight: float = 0.5,
    size_weight: float = 0.25,
    confidence_weight: float = 0.1,
    gap_weight: float = 0.05,
    source_continuity_bonus: float = 0.35,
    tiny_scale: float = 16.0,
    min_candidate_confidence: float = 0.0,
    emit_coasts: bool = False,
    coast_max_gap: int = 2,
    coast_confidence_decay: float = 0.85,
    min_coast_confidence: float = 0.25,
    max_coast_uncertainty: float = 0.75,
    uncertainty_growth: float = 0.15,
    coast_conflict_iou: float = 0.3,
    sequences: Iterable[str] | None = None,
) -> ClosedWorldSummary:
    """Reassign every candidate to a supplied seed identity or suppress it.

    The first-frame labels define the complete identity bank. Later source track
    IDs are treated only as a soft continuity cue, so an upstream ID switch can
    be absorbed when motion and tiny-object-aware geometry support another seed.
    """

    parameters = _validated_parameters(
        max_gap=max_gap,
        max_cost=max_cost,
        max_normalized_distance=max_normalized_distance,
        min_iou=min_iou,
        min_nwd=min_nwd,
        center_weight=center_weight,
        nwd_weight=nwd_weight,
        iou_weight=iou_weight,
        size_weight=size_weight,
        confidence_weight=confidence_weight,
        gap_weight=gap_weight,
        source_continuity_bonus=source_continuity_bonus,
        tiny_scale=tiny_scale,
        min_candidate_confidence=min_candidate_confidence,
        emit_coasts=emit_coasts,
        coast_max_gap=coast_max_gap,
        coast_confidence_decay=coast_confidence_decay,
        min_coast_confidence=min_coast_confidence,
        max_coast_uncertainty=max_coast_uncertainty,
        uncertainty_growth=uncertainty_growth,
        coast_conflict_iou=coast_conflict_iou,
    )
    _validate_paths(prediction_path, first_frame_label_dir, output_dir)
    label_paths = sorted(first_frame_label_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(
            "first-frame label directory contains no .txt files: "
            f"{first_frame_label_dir}"
        )
    predictions = prediction_texts(prediction_path)
    label_names = {f"{path.stem}.txt" for path in label_paths}
    unexpected = sorted(set(predictions) - label_names)
    if unexpected:
        raise ValueError(
            "prediction input contains unknown sequence files: "
            + ", ".join(unexpected)
        )
    requested = set(sequences or ())
    if requested:
        missing = sorted(requested - {path.stem for path in label_paths})
        if missing:
            raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")

    results: list[_SequenceResult] = []
    for label_path in label_paths:
        sequence = label_path.stem
        if requested and sequence not in requested:
            continue
        seeds = parse_detection_text(
            label_path.read_text(encoding="utf-8"), source=str(label_path)
        )
        if any(row.frame_id != 1 for row in seeds):
            raise ValueError(f"{label_path}: expected first-frame-only labels")
        candidates = parse_detection_text(
            predictions.get(f"{sequence}.txt", ""),
            source=f"{prediction_path}:{sequence}.txt",
        )
        results.append(
            _postprocess_sequence(sequence, seeds, candidates, parameters=parameters)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.txt"):
        stale_path.unlink()
    for result in results:
        output_path = output_dir / f"{result.summary.sequence}.txt"
        output_path.write_text(
            "".join(format_detection(row) + "\n" for row in result.rows),
            encoding="utf-8",
        )

    summaries = tuple(result.summary for result in results)
    return ClosedWorldSummary(
        prediction_path=str(prediction_path),
        first_frame_label_dir=str(first_frame_label_dir),
        output_dir=str(output_dir),
        parameters=parameters,
        sequence_count=len(summaries),
        seed_count=sum(row.seed_count for row in summaries),
        input_rows=sum(row.input_rows for row in summaries),
        output_rows=sum(row.output_rows for row in summaries),
        frame_one_matches=sum(row.frame_one_matches for row in summaries),
        matched_candidate_rows=sum(row.matched_candidate_rows for row in summaries),
        dropped_candidate_rows=sum(row.dropped_candidate_rows for row in summaries),
        absorbed_source_switches=sum(
            row.absorbed_source_switches for row in summaries
        ),
        coasted_rows=sum(row.coasted_rows for row in summaries),
        unmatched_track_frames=sum(row.unmatched_track_frames for row in summaries),
        sequences=summaries,
    )


def _validated_parameters(**values: object) -> ClosedWorldParameters:
    max_gap = validate_nonnegative_int(values["max_gap"], name="max_gap")
    coast_max_gap = validate_nonnegative_int(
        values["coast_max_gap"], name="coast_max_gap"
    )
    if bool(values["emit_coasts"]) and coast_max_gap > max_gap:
        raise ValueError("coast_max_gap cannot exceed max_gap when coasting is enabled")
    tiny_scale = validate_nonnegative_finite(values["tiny_scale"], name="tiny_scale")
    if tiny_scale <= 0.0:
        raise ValueError("tiny_scale must be positive")
    return ClosedWorldParameters(
        max_gap=max_gap,
        max_cost=validate_nonnegative_finite(values["max_cost"], name="max_cost"),
        max_normalized_distance=validate_nonnegative_finite(
            values["max_normalized_distance"], name="max_normalized_distance"
        ),
        min_iou=validate_unit_interval(values["min_iou"], name="min_iou"),
        min_nwd=validate_unit_interval(values["min_nwd"], name="min_nwd"),
        center_weight=validate_nonnegative_finite(
            values["center_weight"], name="center_weight"
        ),
        nwd_weight=validate_nonnegative_finite(
            values["nwd_weight"], name="nwd_weight"
        ),
        iou_weight=validate_nonnegative_finite(
            values["iou_weight"], name="iou_weight"
        ),
        size_weight=validate_nonnegative_finite(
            values["size_weight"], name="size_weight"
        ),
        confidence_weight=validate_nonnegative_finite(
            values["confidence_weight"], name="confidence_weight"
        ),
        gap_weight=validate_nonnegative_finite(
            values["gap_weight"], name="gap_weight"
        ),
        source_continuity_bonus=validate_nonnegative_finite(
            values["source_continuity_bonus"], name="source_continuity_bonus"
        ),
        tiny_scale=tiny_scale,
        min_candidate_confidence=validate_nonnegative_finite(
            values["min_candidate_confidence"], name="min_candidate_confidence"
        ),
        emit_coasts=bool(values["emit_coasts"]),
        coast_max_gap=coast_max_gap,
        coast_confidence_decay=validate_unit_interval(
            values["coast_confidence_decay"], name="coast_confidence_decay"
        ),
        min_coast_confidence=validate_nonnegative_finite(
            values["min_coast_confidence"], name="min_coast_confidence"
        ),
        max_coast_uncertainty=validate_nonnegative_finite(
            values["max_coast_uncertainty"], name="max_coast_uncertainty"
        ),
        uncertainty_growth=validate_nonnegative_finite(
            values["uncertainty_growth"], name="uncertainty_growth"
        ),
        coast_conflict_iou=validate_unit_interval(
            values["coast_conflict_iou"], name="coast_conflict_iou"
        ),
    )


def _validate_paths(
    prediction_path: Path, first_frame_label_dir: Path, output_dir: Path
) -> None:
    if not first_frame_label_dir.exists():
        raise FileNotFoundError(
            f"first-frame label directory does not exist: {first_frame_label_dir}"
        )
    if not first_frame_label_dir.is_dir():
        raise NotADirectoryError(
            f"first-frame label path is not a directory: {first_frame_label_dir}"
        )
    output_resolved = output_dir.resolve()
    if output_resolved == first_frame_label_dir.resolve():
        raise ValueError(
            "output directory must differ from first-frame label directory: "
            f"{output_dir}"
        )
    if prediction_path.is_dir() and output_resolved == prediction_path.resolve():
        raise ValueError(
            f"output directory must differ from prediction directory: {output_dir}"
        )


def _postprocess_sequence(
    sequence: str,
    seeds: list[Detection],
    candidates: list[Detection],
    *,
    parameters: ClosedWorldParameters,
) -> _SequenceResult:
    reject_duplicate_keys(seeds, label="seed")
    reject_duplicate_keys(candidates, label="prediction")
    seed_rows = tuple(sorted(seeds, key=lambda row: row.object_id))
    candidate_frames = rows_by_frame(candidates)
    states = [_TrackState(row.object_id, [row]) for row in seed_rows]
    output_rows: list[Detection] = list(seed_rows)

    frame_one_candidates = tuple(
        row
        for row in candidate_frames.get(1, ())
        if row.confidence >= parameters.min_candidate_confidence
    )
    initial_matches = _assign_candidates(
        states,
        frame_one_candidates,
        frame_id=1,
        parameters=parameters,
        initial=True,
    )
    for track_index, candidate_index in initial_matches.items():
        states[track_index].last_source_id = frame_one_candidates[
            candidate_index
        ].object_id

    matched_rows = 0
    source_switches = 0
    coasted_rows = 0
    unmatched_track_frames = 0
    max_frame = max([1, *candidate_frames])
    for frame_id in range(2, max_frame + 1):
        frame_candidates = tuple(
            row
            for row in candidate_frames.get(frame_id, ())
            if row.confidence >= parameters.min_candidate_confidence
        )
        active_indices = [
            index
            for index, state in enumerate(states)
            if frame_id - state.last_observation.frame_id - 1 <= parameters.max_gap
        ]
        active_states = [states[index] for index in active_indices]
        local_matches = _assign_candidates(
            active_states,
            frame_candidates,
            frame_id=frame_id,
            parameters=parameters,
            initial=False,
        )
        matched_global_indices: set[int] = set()
        accepted_this_frame: list[Detection] = []
        for local_track_index, candidate_index in sorted(local_matches.items()):
            global_track_index = active_indices[local_track_index]
            state = states[global_track_index]
            candidate = frame_candidates[candidate_index]
            previous_source_id = state.last_source_id
            if previous_source_id is not None and candidate.object_id != previous_source_id:
                source_switches += 1
            assigned = replace(candidate, object_id=state.seed_id)
            state.observations.append(assigned)
            state.last_source_id = candidate.object_id
            output_rows.append(assigned)
            accepted_this_frame.append(assigned)
            matched_rows += 1
            matched_global_indices.add(global_track_index)

        unmatched_track_frames += len(active_indices) - len(matched_global_indices)
        if parameters.emit_coasts:
            for global_track_index in active_indices:
                if global_track_index in matched_global_indices:
                    continue
                coasted = _coast_row(
                    states[global_track_index],
                    frame_id=frame_id,
                    occupied=tuple(accepted_this_frame),
                    parameters=parameters,
                )
                if coasted is None:
                    continue
                output_rows.append(coasted)
                accepted_this_frame.append(coasted)
                coasted_rows += 1

    output = tuple(sorted(output_rows, key=lambda row: (row.frame_id, row.object_id)))
    frame_one_match_count = len(initial_matches)
    matched_candidate_count = frame_one_match_count + matched_rows
    return _SequenceResult(
        output,
        SequenceClosedWorldSummary(
            sequence=sequence,
            seed_count=len(seed_rows),
            input_rows=len(candidates),
            output_rows=len(output),
            max_frame=max_frame,
            frame_one_matches=frame_one_match_count,
            matched_candidate_rows=matched_candidate_count,
            dropped_candidate_rows=len(candidates) - matched_candidate_count,
            absorbed_source_switches=source_switches,
            coasted_rows=coasted_rows,
            unmatched_track_frames=unmatched_track_frames,
        ),
    )


def _assign_candidates(
    states: list[_TrackState],
    candidates: tuple[Detection, ...],
    *,
    frame_id: int,
    parameters: ClosedWorldParameters,
    initial: bool,
) -> dict[int, int]:
    if not states or not candidates:
        return {}
    costs = np.full((len(states), len(candidates)), _INVALID_COST, dtype=float)
    for track_index, state in enumerate(states):
        for candidate_index, candidate in enumerate(candidates):
            cost = _association_cost(
                state,
                candidate,
                frame_id=frame_id,
                parameters=parameters,
                initial=initial,
            )
            if cost is not None and cost <= parameters.max_cost:
                costs[track_index, candidate_index] = cost

    unmatched_cost = parameters.max_cost + 1e-6
    augmented = np.full(
        (len(states), len(candidates) + len(states)),
        unmatched_cost,
        dtype=float,
    )
    augmented[:, : len(candidates)] = costs
    augmented += np.arange(augmented.shape[1], dtype=float)[None, :] * 1e-12
    rows, cols = linear_sum_assignment(augmented)
    return {
        int(track_index): int(candidate_index)
        for track_index, candidate_index in zip(rows, cols, strict=True)
        if candidate_index < len(candidates)
        and costs[track_index, candidate_index] < _INVALID_COST
    }


def _association_cost(
    state: _TrackState,
    candidate: Detection,
    *,
    frame_id: int,
    parameters: ClosedWorldParameters,
    initial: bool,
) -> float | None:
    last = state.last_observation
    if candidate.class_id != last.class_id:
        return None
    predicted = last if initial else _predict_box(state.observations, frame_id=frame_id)
    gap = 0 if initial else frame_id - last.frame_id - 1
    center_distance = _normalized_center_distance(predicted, candidate)
    distance_gate = parameters.max_normalized_distance * math.sqrt(gap + 1.0)
    if center_distance > distance_gate:
        return None
    iou = box_iou(predicted, candidate)
    if iou < parameters.min_iou:
        return None
    nwd = _nwd_similarity(predicted, candidate)
    if nwd < parameters.min_nwd:
        return None
    scale_change = abs(math.log(candidate.width / predicted.width)) + abs(
        math.log(candidate.height / predicted.height)
    )
    target_scale = math.sqrt(
        max(
            _EPS,
            0.5
            * (
                predicted.width * predicted.height
                + candidate.width * candidate.height
            ),
        )
    )
    iou_reliability = min(1.0, target_scale / parameters.tiny_scale)
    confidence = min(1.0, max(0.0, candidate.confidence))
    cost = (
        parameters.center_weight * center_distance
        + parameters.nwd_weight * (1.0 - nwd)
        + parameters.iou_weight * iou_reliability * (1.0 - iou)
        + parameters.size_weight * scale_change
        + parameters.confidence_weight * (1.0 - confidence)
        + parameters.gap_weight * gap
    )
    if not initial and state.last_source_id == candidate.object_id:
        cost -= parameters.source_continuity_bonus
    return cost


def _predict_box(observations: list[Detection], *, frame_id: int) -> Detection:
    last = observations[-1]
    delta = frame_id - last.frame_id
    if len(observations) < 2 or delta <= 0:
        return replace(last, frame_id=frame_id)
    previous = observations[-2]
    history_delta = last.frame_id - previous.frame_id
    if history_delta <= 0:
        return replace(last, frame_id=frame_id)
    velocity_x = (last.center_x - previous.center_x) / history_delta
    velocity_y = (last.center_y - previous.center_y) / history_delta
    log_width_rate = math.log(last.width / previous.width) / history_delta
    log_height_rate = math.log(last.height / previous.height) / history_delta
    width = max(_EPS, last.width * math.exp(log_width_rate * delta))
    height = max(_EPS, last.height * math.exp(log_height_rate * delta))
    center_x = last.center_x + velocity_x * delta
    center_y = last.center_y + velocity_y * delta
    return replace(
        last,
        frame_id=frame_id,
        x1=center_x - 0.5 * width,
        y1=center_y - 0.5 * height,
        width=width,
        height=height,
    )


def _normalized_center_distance(left: Detection, right: Detection) -> float:
    scale = max(
        1.0,
        math.sqrt(max(_EPS, left.width * left.height)),
        math.sqrt(max(_EPS, right.width * right.height)),
    )
    return (
        math.hypot(left.center_x - right.center_x, left.center_y - right.center_y)
        / scale
    )


def _nwd_similarity(left: Detection, right: Detection) -> float:
    scale = max(
        1.0,
        math.sqrt(
            max(
                _EPS,
                0.5 * (left.width * left.height + right.width * right.height),
            )
        ),
    )
    distance = (
        math.sqrt(
            (left.center_x - right.center_x) ** 2
            + (left.center_y - right.center_y) ** 2
            + 0.25 * (left.width - right.width) ** 2
            + 0.25 * (left.height - right.height) ** 2
        )
        / scale
    )
    return math.exp(-distance)


def _coast_row(
    state: _TrackState,
    *,
    frame_id: int,
    occupied: tuple[Detection, ...],
    parameters: ClosedWorldParameters,
) -> Detection | None:
    last = state.last_observation
    gap = frame_id - last.frame_id
    if gap <= 0 or gap > parameters.coast_max_gap:
        return None
    uncertainty = _motion_uncertainty(state.observations) + (
        parameters.uncertainty_growth * gap
    )
    if uncertainty > parameters.max_coast_uncertainty:
        return None
    confidence = last.confidence * parameters.coast_confidence_decay**gap
    if confidence < parameters.min_coast_confidence:
        return None
    predicted = replace(
        _predict_box(state.observations, frame_id=frame_id),
        object_id=state.seed_id,
        confidence=confidence,
        visibility=min(
            1.0,
            max(
                0.0,
                last.visibility * parameters.coast_confidence_decay**gap,
            ),
        ),
    )
    if any(
        box_iou(predicted, row) > parameters.coast_conflict_iou for row in occupied
    ):
        return None
    return predicted


def _motion_uncertainty(observations: list[Detection]) -> float:
    if len(observations) < 3:
        return 0.0
    predicted = _predict_box(observations[:-1], frame_id=observations[-1].frame_id)
    actual = observations[-1]
    center_residual = _normalized_center_distance(predicted, actual)
    size_residual = abs(math.log(actual.width / predicted.width)) + abs(
        math.log(actual.height / predicted.height)
    )
    return center_residual + 0.25 * size_residual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-gap", type=int, default=15)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--max-normalized-distance", type=float, default=4.0)
    parser.add_argument("--min-iou", type=float, default=0.0)
    parser.add_argument("--min-nwd", type=float, default=0.0)
    parser.add_argument("--source-continuity-bonus", type=float, default=0.35)
    parser.add_argument("--min-candidate-confidence", type=float, default=0.0)
    parser.add_argument("--emit-coasts", action="store_true")
    parser.add_argument("--coast-max-gap", type=int, default=2)
    parser.add_argument("--coast-confidence-decay", type=float, default=0.85)
    parser.add_argument("--min-coast-confidence", type=float, default=0.25)
    parser.add_argument("--max-coast-uncertainty", type=float, default=0.75)
    parser.add_argument("--sequences", nargs="*", default=[])
    args = parser.parse_args(argv)
    summary = postprocess_closed_world(
        args.prediction_path,
        args.first_frame_label_dir,
        args.output_dir,
        max_gap=args.max_gap,
        max_cost=args.max_cost,
        max_normalized_distance=args.max_normalized_distance,
        min_iou=args.min_iou,
        min_nwd=args.min_nwd,
        source_continuity_bonus=args.source_continuity_bonus,
        min_candidate_confidence=args.min_candidate_confidence,
        emit_coasts=args.emit_coasts,
        coast_max_gap=args.coast_max_gap,
        coast_confidence_decay=args.coast_confidence_decay,
        min_coast_confidence=args.min_coast_confidence,
        max_coast_uncertainty=args.max_coast_uncertainty,
        sequences=tuple(args.sequences),
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(f"sequence_count={summary.sequence_count}")
    print(f"matched_candidate_rows={summary.matched_candidate_rows}")
    print(f"absorbed_source_switches={summary.absorbed_source_switches}")
    print(f"coasted_rows={summary.coasted_rows}")
    print(f"dropped_candidate_rows={summary.dropped_candidate_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
