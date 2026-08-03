"""Beam/MHT fixed-population post-processing for Multi-UAV LTS predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from ._records import (
    Detection,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)
from .fixed_population import (
    _deduplicate,
    _interpolate_single_frame_gaps,
    _link_cost,
    _reject_output_input_aliases,
    _seed_track_mapping,
    _split_tracklets,
)
from .mht_relink import relink_tracklets_beam


@dataclass(frozen=True)
class SequenceBeamPostprocessSummary:
    sequence: str
    seed_count: int
    input_rows: int
    output_rows: int
    mapped_input_tracks: int
    dropped_input_tracks: int
    relinked_tracklets: int
    inserted_seed_rows: int
    interpolated_rows: int
    evaluated_hypotheses: int
    best_hypothesis_cost: float
    second_best_margin: float | None


@dataclass(frozen=True)
class BeamFixedPopulationSummary:
    prediction_path: str
    first_frame_label_dir: str
    output_dir: str
    min_seed_iou: float
    relink_max_gap: int
    relink_max_cost: float
    relink_beam_width: int
    relink_drop_cost: float
    relink_velocity_weight: float
    interpolate_single_frame: bool
    sequence_count: int
    input_rows: int
    output_rows: int
    mapped_input_tracks: int
    dropped_input_tracks: int
    relinked_tracklets: int
    inserted_seed_rows: int
    interpolated_rows: int
    evaluated_hypotheses: int
    sequences: tuple[SequenceBeamPostprocessSummary, ...]


def postprocess_fixed_population_beam(
    prediction_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    min_seed_iou: float = 0.5,
    relink_max_gap: int = 5,
    relink_max_cost: float = 1.5,
    relink_beam_width: int = 16,
    relink_drop_cost: float | None = None,
    relink_velocity_weight: float = 0.25,
    interpolate_single_frame: bool = False,
    sequences: Iterable[str] | None = None,
) -> BeamFixedPopulationSummary:
    """Map tracks to seed IDs and globally relink fragments with a bounded beam."""

    min_iou = validate_unit_interval(min_seed_iou, name="min_seed_iou")
    max_gap = validate_nonnegative_int(relink_max_gap, name="relink_max_gap")
    max_cost = validate_nonnegative_finite(relink_max_cost, name="relink_max_cost")
    beam_width = validate_nonnegative_int(
        relink_beam_width, name="relink_beam_width"
    )
    if beam_width <= 0:
        raise ValueError("relink_beam_width must be a positive integer")
    drop_cost = (
        max_cost
        if relink_drop_cost is None
        else validate_nonnegative_finite(relink_drop_cost, name="relink_drop_cost")
    )
    velocity_weight = validate_nonnegative_finite(
        relink_velocity_weight, name="relink_velocity_weight"
    )
    _validate_inputs(prediction_path, first_frame_label_dir, output_dir)

    label_paths = sorted(first_frame_label_dir.glob("*.txt"))
    predictions = prediction_texts(prediction_path)
    requested = set(sequences or ())
    if requested:
        missing = sorted(requested - {path.stem for path in label_paths})
        if missing:
            raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")

    results: list[tuple[tuple[Detection, ...], SequenceBeamPostprocessSummary]] = []
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
        results.append(
            _postprocess_sequence_beam(
                sequence,
                seeds,
                rows,
                min_seed_iou=min_iou,
                relink_max_gap=max_gap,
                relink_max_cost=max_cost,
                relink_beam_width=beam_width,
                relink_drop_cost=drop_cost,
                relink_velocity_weight=velocity_weight,
                interpolate_single_frame=interpolate_single_frame,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.txt"):
        stale_path.unlink()
    for rows, summary in results:
        (output_dir / f"{summary.sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    summaries = tuple(summary for _, summary in results)
    return BeamFixedPopulationSummary(
        prediction_path=str(prediction_path),
        first_frame_label_dir=str(first_frame_label_dir),
        output_dir=str(output_dir),
        min_seed_iou=min_iou,
        relink_max_gap=max_gap,
        relink_max_cost=max_cost,
        relink_beam_width=beam_width,
        relink_drop_cost=drop_cost,
        relink_velocity_weight=velocity_weight,
        interpolate_single_frame=bool(interpolate_single_frame),
        sequence_count=len(summaries),
        input_rows=sum(row.input_rows for row in summaries),
        output_rows=sum(row.output_rows for row in summaries),
        mapped_input_tracks=sum(row.mapped_input_tracks for row in summaries),
        dropped_input_tracks=sum(row.dropped_input_tracks for row in summaries),
        relinked_tracklets=sum(row.relinked_tracklets for row in summaries),
        inserted_seed_rows=sum(row.inserted_seed_rows for row in summaries),
        interpolated_rows=sum(row.interpolated_rows for row in summaries),
        evaluated_hypotheses=sum(row.evaluated_hypotheses for row in summaries),
        sequences=summaries,
    )


def _validate_inputs(
    prediction_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
) -> None:
    if not first_frame_label_dir.exists():
        raise FileNotFoundError(
            f"first-frame label directory does not exist: {first_frame_label_dir}"
        )
    if not first_frame_label_dir.is_dir():
        raise NotADirectoryError(
            f"first-frame label path is not a directory: {first_frame_label_dir}"
        )
    _reject_output_input_aliases(prediction_path, first_frame_label_dir, output_dir)
    if not any(first_frame_label_dir.glob("*.txt")):
        raise ValueError(
            f"first-frame label directory contains no .txt files: {first_frame_label_dir}"
        )


def _postprocess_sequence_beam(
    sequence: str,
    seeds: list[Detection],
    predictions: list[Detection],
    *,
    min_seed_iou: float,
    relink_max_gap: int,
    relink_max_cost: float,
    relink_beam_width: int,
    relink_drop_cost: float,
    relink_velocity_weight: float,
    interpolate_single_frame: bool,
) -> tuple[tuple[Detection, ...], SequenceBeamPostprocessSummary]:
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

    tracklets = [
        tracklet
        for tracklet in _split_tracklets(predictions)
        if tracklet.source_id not in mapping and tracklet.start_frame > 1
    ]

    def beam_link_cost(
        path: list[Detection], candidate: tuple[Detection, ...], *, gap: int
    ) -> float:
        return _link_cost(path, candidate, gap=gap) + (
            relink_velocity_weight * _velocity_disagreement(path, candidate)
        )

    relink = relink_tracklets_beam(
        assigned,
        tracklets,
        max_gap=relink_max_gap,
        max_cost=relink_max_cost,
        beam_width=relink_beam_width,
        drop_cost=relink_drop_cost,
        link_cost=beam_link_cost,
    )
    assigned_rows = {seed_id: list(rows) for seed_id, rows in relink.assigned.items()}
    output_rows = _deduplicate(assigned_rows)
    interpolated_count = 0
    if interpolate_single_frame:
        output_rows, interpolated_count = _interpolate_single_frame_gaps(output_rows)
    output_rows = tuple(
        sorted(output_rows, key=lambda row: (row.frame_id, row.object_id))
    )
    source_ids = {row.object_id for row in predictions}
    dropped = len(source_ids - set(mapping) - set(relink.relinked_source_ids))
    mapped_seed_ids = set(mapping.values())
    return output_rows, SequenceBeamPostprocessSummary(
        sequence=sequence,
        seed_count=len(seed_rows),
        input_rows=len(predictions),
        output_rows=len(output_rows),
        mapped_input_tracks=len(mapping),
        dropped_input_tracks=dropped,
        relinked_tracklets=relink.relinked_tracklets,
        inserted_seed_rows=len({row.object_id for row in seed_rows} - mapped_seed_ids),
        interpolated_rows=interpolated_count,
        evaluated_hypotheses=relink.evaluated_hypotheses,
        best_hypothesis_cost=relink.best_cost,
        second_best_margin=relink.second_best_margin,
    )


def _velocity_disagreement(
    path: list[Detection], candidate: tuple[Detection, ...]
) -> float:
    if len(path) < 2 or len(candidate) < 2:
        return 0.0
    previous, last = path[-2], path[-1]
    path_delta = last.frame_id - previous.frame_id
    first, candidate_last = candidate[0], candidate[-1]
    candidate_delta = candidate_last.frame_id - first.frame_id
    if path_delta <= 0 or candidate_delta <= 0:
        return 0.0
    path_velocity_x = (last.center_x - previous.center_x) / path_delta
    path_velocity_y = (last.center_y - previous.center_y) / path_delta
    candidate_velocity_x = (candidate_last.center_x - first.center_x) / candidate_delta
    candidate_velocity_y = (candidate_last.center_y - first.center_y) / candidate_delta
    scale = max(
        1.0,
        (last.width * last.height) ** 0.5,
        (first.width * first.height) ** 0.5,
    )
    return (
        (path_velocity_x - candidate_velocity_x) ** 2
        + (path_velocity_y - candidate_velocity_y) ** 2
    ) ** 0.5 / scale


def write_summary_json(summary: BeamFixedPopulationSummary, path: Path) -> None:
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
    parser.add_argument("--relink-max-gap", type=int, default=5)
    parser.add_argument("--relink-max-cost", type=float, default=1.5)
    parser.add_argument("--relink-beam-width", type=int, default=16)
    parser.add_argument("--relink-drop-cost", type=float)
    parser.add_argument("--relink-velocity-weight", type=float, default=0.25)
    parser.add_argument("--interpolate-single-frame", action="store_true")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    summary = postprocess_fixed_population_beam(
        args.prediction_path,
        args.first_frame_label_dir,
        args.output_dir,
        min_seed_iou=args.min_seed_iou,
        relink_max_gap=args.relink_max_gap,
        relink_max_cost=args.relink_max_cost,
        relink_beam_width=args.relink_beam_width,
        relink_drop_cost=args.relink_drop_cost,
        relink_velocity_weight=args.relink_velocity_weight,
        interpolate_single_frame=args.interpolate_single_frame,
        sequences=args.sequences,
    )
    if args.output_json:
        write_summary_json(summary, args.output_json)
    print(f"fixed_population_beam_sequences={summary.sequence_count}")
    print(f"fixed_population_beam_output_rows={summary.output_rows}")
    print(f"fixed_population_beam_dropped_tracks={summary.dropped_input_tracks}")
    print(f"fixed_population_beam_relinked_tracklets={summary.relinked_tracklets}")
    print(f"fixed_population_beam_hypotheses={summary.evaluated_hypotheses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
