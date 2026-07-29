"""First-frame-seeded post-processing for Multi-UAV LTS predictions."""

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
    iou_matrix,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)

_EPS = 1e-12


@dataclass(frozen=True)
class SequencePostprocessSummary:
    sequence: str
    seed_count: int
    input_rows: int
    output_rows: int
    mapped_input_tracks: int
    dropped_input_tracks: int
    relinked_tracklets: int
    inserted_seed_rows: int
    interpolated_rows: int


@dataclass(frozen=True)
class FixedPopulationSummary:
    prediction_path: str
    first_frame_label_dir: str
    output_dir: str
    min_seed_iou: float
    relink_max_gap: int
    relink_max_cost: float
    interpolate_single_frame: bool
    sequence_count: int
    input_rows: int
    output_rows: int
    mapped_input_tracks: int
    dropped_input_tracks: int
    relinked_tracklets: int
    inserted_seed_rows: int
    interpolated_rows: int
    sequences: tuple[SequencePostprocessSummary, ...]


@dataclass(frozen=True)
class _Tracklet:
    source_id: int
    index: int
    rows: tuple[Detection, ...]

    @property
    def start_frame(self) -> int:
        return self.rows[0].frame_id


@dataclass(frozen=True)
class _SequenceResult:
    rows: tuple[Detection, ...]
    summary: SequencePostprocessSummary


def postprocess_fixed_population(
    prediction_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    min_seed_iou: float = 0.5,
    relink_max_gap: int = 0,
    relink_max_cost: float = 2.0,
    interpolate_single_frame: bool = False,
    sequences: Iterable[str] | None = None,
) -> FixedPopulationSummary:
    """Map tracks to supplied seed IDs, reject births, and reconnect fragments."""

    min_iou = validate_unit_interval(min_seed_iou, name="min_seed_iou")
    max_gap = validate_nonnegative_int(relink_max_gap, name="relink_max_gap")
    max_cost = validate_nonnegative_finite(relink_max_cost, name="relink_max_cost")
    predictions = prediction_texts(prediction_path)
    requested = set(sequences or ())
    label_paths = sorted(first_frame_label_dir.glob("*.txt"))
    if requested:
        missing = sorted(requested - {path.stem for path in label_paths})
        if missing:
            raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[SequencePostprocessSummary] = []
    for label_path in label_paths:
        sequence = label_path.stem
        if requested and sequence not in requested:
            continue
        seeds = parse_detection_text(
            label_path.read_text(encoding="utf-8"), source=str(label_path)
        )
        if any(row.frame_id != 1 for row in seeds):
            raise ValueError(f"{label_path}: expected first-frame-only labels")
        rows = parse_detection_text(
            predictions.get(f"{sequence}.txt", ""),
            source=f"{prediction_path}:{sequence}.txt",
        )
        result = _postprocess_sequence(
            sequence,
            seeds,
            rows,
            min_seed_iou=min_iou,
            relink_max_gap=max_gap,
            relink_max_cost=max_cost,
            interpolate_single_frame=interpolate_single_frame,
        )
        (output_dir / f"{sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in result.rows),
            encoding="utf-8",
        )
        summaries.append(result.summary)

    return FixedPopulationSummary(
        prediction_path=str(prediction_path),
        first_frame_label_dir=str(first_frame_label_dir),
        output_dir=str(output_dir),
        min_seed_iou=min_iou,
        relink_max_gap=max_gap,
        relink_max_cost=max_cost,
        interpolate_single_frame=bool(interpolate_single_frame),
        sequence_count=len(summaries),
        input_rows=sum(row.input_rows for row in summaries),
        output_rows=sum(row.output_rows for row in summaries),
        mapped_input_tracks=sum(row.mapped_input_tracks for row in summaries),
        dropped_input_tracks=sum(row.dropped_input_tracks for row in summaries),
        relinked_tracklets=sum(row.relinked_tracklets for row in summaries),
        inserted_seed_rows=sum(row.inserted_seed_rows for row in summaries),
        interpolated_rows=sum(row.interpolated_rows for row in summaries),
        sequences=tuple(summaries),
    )


def _postprocess_sequence(
    sequence: str,
    seeds: list[Detection],
    predictions: list[Detection],
    *,
    min_seed_iou: float,
    relink_max_gap: int,
    relink_max_cost: float,
    interpolate_single_frame: bool,
) -> _SequenceResult:
    reject_duplicate_keys(seeds, label="seed")
    reject_duplicate_keys(predictions, label="prediction")
    seed_rows = tuple(sorted(seeds, key=lambda row: row.object_id))
    frame_one = tuple(
        sorted(
            (row for row in predictions if row.frame_id == 1),
            key=lambda row: row.object_id,
        )
    )
    mapping = _seed_track_mapping(seed_rows, frame_one, min_iou=min_seed_iou)
    assigned = {seed.object_id: [seed] for seed in seed_rows}
    for row in predictions:
        seed_id = mapping.get(row.object_id)
        if seed_id is not None and row.frame_id != 1:
            assigned[seed_id].append(replace(row, object_id=seed_id))

    relinked_count = 0
    relinked_source_ids: set[int] = set()
    if relink_max_gap > 0 and assigned:
        tracklets = [
            tracklet
            for tracklet in _split_tracklets(predictions)
            if tracklet.source_id not in mapping and tracklet.start_frame > 1
        ]
        relinked_count, relinked_source_ids = _relink_tracklets(
            assigned,
            tracklets,
            max_gap=relink_max_gap,
            max_cost=relink_max_cost,
        )

    output_rows = _deduplicate(assigned)
    interpolated_count = 0
    if interpolate_single_frame:
        output_rows, interpolated_count = _interpolate_single_frame_gaps(output_rows)
    output_rows = tuple(sorted(output_rows, key=lambda row: (row.frame_id, row.object_id)))
    source_ids = {row.object_id for row in predictions}
    dropped = len(source_ids - set(mapping) - relinked_source_ids)
    mapped_seed_ids = set(mapping.values())
    return _SequenceResult(
        output_rows,
        SequencePostprocessSummary(
            sequence=sequence,
            seed_count=len(seed_rows),
            input_rows=len(predictions),
            output_rows=len(output_rows),
            mapped_input_tracks=len(mapping),
            dropped_input_tracks=dropped,
            relinked_tracklets=relinked_count,
            inserted_seed_rows=len({row.object_id for row in seed_rows} - mapped_seed_ids),
            interpolated_rows=interpolated_count,
        ),
    )


def _seed_track_mapping(
    seeds: tuple[Detection, ...],
    frame_one: tuple[Detection, ...],
    *,
    min_iou: float,
) -> dict[int, int]:
    if not seeds or not frame_one:
        return {}
    overlaps = iou_matrix(seeds, frame_one)
    valid = overlaps >= min_iou
    # Prioritize the number of gate-valid pairs before using IoU as a tie-breaker.
    cardinality_bonus = float(min(overlaps.shape) + 1)
    score = np.where(valid, cardinality_bonus + overlaps, 0.0)
    rows, cols = linear_sum_assignment(-score)
    return {
        frame_one[prediction_index].object_id: seeds[seed_index].object_id
        for seed_index, prediction_index in zip(rows, cols, strict=True)
        if valid[seed_index, prediction_index]
    }


def _split_tracklets(rows: list[Detection]) -> list[_Tracklet]:
    by_id: dict[int, list[Detection]] = {}
    for row in rows:
        by_id.setdefault(row.object_id, []).append(row)
    result: list[_Tracklet] = []
    for source_id, source_rows in sorted(by_id.items()):
        source_rows.sort(key=lambda row: row.frame_id)
        current: list[Detection] = []
        index = 0
        previous_frame: int | None = None
        for row in source_rows:
            if previous_frame is not None and row.frame_id != previous_frame + 1:
                result.append(_Tracklet(source_id, index, tuple(current)))
                current = []
                index += 1
            current.append(row)
            previous_frame = row.frame_id
        if current:
            result.append(_Tracklet(source_id, index, tuple(current)))
    return result


def _relink_tracklets(
    assigned: dict[int, list[Detection]],
    tracklets: list[_Tracklet],
    *,
    max_gap: int,
    max_cost: float,
) -> tuple[int, set[int]]:
    relinked = 0
    source_ids: set[int] = set()
    by_start: dict[int, list[_Tracklet]] = {}
    for tracklet in tracklets:
        by_start.setdefault(tracklet.start_frame, []).append(tracklet)
    for start_frame in sorted(by_start):
        candidates = sorted(by_start[start_frame], key=lambda item: (item.source_id, item.index))
        seed_ids = sorted(assigned)
        costs = np.full((len(seed_ids), len(candidates)), np.inf, dtype=float)
        for seed_index, seed_id in enumerate(seed_ids):
            path = sorted(assigned[seed_id], key=lambda row: row.frame_id)
            for tracklet_index, tracklet in enumerate(candidates):
                if _has_overlap(path, tracklet.rows):
                    continue
                gap = tracklet.start_frame - path[-1].frame_id - 1
                if 0 <= gap <= max_gap:
                    costs[seed_index, tracklet_index] = _link_cost(
                        path, tracklet.rows, gap=gap
                    )
        if not np.isfinite(costs).any():
            continue
        rows, cols = linear_sum_assignment(
            np.where(np.isfinite(costs), costs, max_cost + 1e6)
        )
        for seed_index, tracklet_index in zip(rows, cols, strict=True):
            if not math.isfinite(costs[seed_index, tracklet_index]):
                continue
            if costs[seed_index, tracklet_index] > max_cost:
                continue
            seed_id = seed_ids[seed_index]
            tracklet = candidates[tracklet_index]
            assigned[seed_id].extend(
                replace(row, object_id=seed_id) for row in tracklet.rows
            )
            assigned[seed_id].sort(key=lambda row: row.frame_id)
            relinked += 1
            source_ids.add(tracklet.source_id)
    return relinked, source_ids


def _link_cost(
    path: list[Detection], candidate: tuple[Detection, ...], *, gap: int
) -> float:
    first = candidate[0]
    predicted = _predict_box(path, frame_id=first.frame_id)
    scale = max(
        1.0,
        math.sqrt(max(_EPS, predicted.width * predicted.height)),
        math.sqrt(max(_EPS, first.width * first.height)),
    )
    center_distance = math.hypot(
        predicted.center_x - first.center_x,
        predicted.center_y - first.center_y,
    ) / scale
    scale_change = abs(math.log(first.width / predicted.width)) + abs(
        math.log(first.height / predicted.height)
    )
    return (
        center_distance
        + 0.25 * scale_change
        + 0.25 * (1.0 - box_iou(predicted, first))
        + 0.05 * gap
    )


def _predict_box(path: list[Detection], *, frame_id: int) -> Detection:
    last = path[-1]
    delta = frame_id - last.frame_id
    if len(path) < 2 or delta <= 0:
        return replace(last, frame_id=frame_id)
    previous = path[-2]
    history_delta = last.frame_id - previous.frame_id
    if history_delta <= 0:
        return replace(last, frame_id=frame_id)
    velocity_x = (last.center_x - previous.center_x) / history_delta
    velocity_y = (last.center_y - previous.center_y) / history_delta
    center_x = last.center_x + velocity_x * delta
    center_y = last.center_y + velocity_y * delta
    return replace(
        last,
        frame_id=frame_id,
        x1=center_x - 0.5 * last.width,
        y1=center_y - 0.5 * last.height,
    )


def _has_overlap(path: list[Detection], rows: tuple[Detection, ...]) -> bool:
    occupied = {row.frame_id for row in path}
    return any(row.frame_id in occupied for row in rows)


def _deduplicate(assigned: dict[int, list[Detection]]) -> tuple[Detection, ...]:
    output: list[Detection] = []
    for seed_id, rows in assigned.items():
        by_frame: dict[int, Detection] = {}
        for row in rows:
            candidate = replace(row, object_id=seed_id)
            current = by_frame.get(row.frame_id)
            if current is None or candidate.confidence > current.confidence:
                by_frame[row.frame_id] = candidate
        output.extend(by_frame.values())
    return tuple(output)


def _interpolate_single_frame_gaps(
    rows: tuple[Detection, ...],
) -> tuple[tuple[Detection, ...], int]:
    by_id: dict[int, list[Detection]] = {}
    for row in rows:
        by_id.setdefault(row.object_id, []).append(row)
    output = list(rows)
    inserted = 0
    for track_rows in by_id.values():
        track_rows.sort(key=lambda row: row.frame_id)
        for left, right in zip(track_rows, track_rows[1:]):
            if right.frame_id - left.frame_id != 2:
                continue
            output.append(
                Detection(
                    left.frame_id + 1,
                    left.object_id,
                    0.5 * (left.x1 + right.x1),
                    0.5 * (left.y1 + right.y1),
                    0.5 * (left.width + right.width),
                    0.5 * (left.height + right.height),
                    min(left.confidence, right.confidence),
                    left.class_id,
                    min(left.visibility, right.visibility),
                )
            )
            inserted += 1
    return tuple(output), inserted


def write_summary_json(summary: FixedPopulationSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-seed-iou", type=float, default=0.5)
    parser.add_argument("--relink-max-gap", type=int, default=0)
    parser.add_argument("--relink-max-cost", type=float, default=2.0)
    parser.add_argument("--interpolate-single-frame", action="store_true")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    summary = postprocess_fixed_population(
        args.prediction_path,
        args.first_frame_label_dir,
        args.output_dir,
        min_seed_iou=args.min_seed_iou,
        relink_max_gap=args.relink_max_gap,
        relink_max_cost=args.relink_max_cost,
        interpolate_single_frame=args.interpolate_single_frame,
        sequences=args.sequences,
    )
    if args.output_json:
        write_summary_json(summary, args.output_json)
    print(f"fixed_population_sequences={summary.sequence_count}")
    print(f"fixed_population_output_rows={summary.output_rows}")
    print(f"fixed_population_dropped_tracks={summary.dropped_input_tracks}")
    print(f"fixed_population_relinked_tracklets={summary.relinked_tracklets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
