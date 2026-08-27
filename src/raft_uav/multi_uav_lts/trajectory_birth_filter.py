"""Truth-free pruning of implausible late-birth tracks in LTS predictions."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ._records import (
    Detection,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
)

_SCHEMA = "raft-uav-multi-uav-lts-birth-filter-v1"


@dataclass(frozen=True)
class BirthFilterParameters:
    """Inference-only controls for retaining an unseeded identity."""

    min_hits: int = 5
    min_span: int = 4
    min_mean_confidence: float = 0.01
    require_border_entry: bool = False
    min_inward_motion: float = 0.0
    image_width: float | None = None
    image_height: float | None = None
    border_margin_fraction: float = 0.08
    drop_all_births: bool = False

    def validate(self) -> None:
        if isinstance(self.min_hits, bool) or self.min_hits <= 0:
            raise ValueError("min_hits must be a positive integer")
        if isinstance(self.min_span, bool) or self.min_span < 0:
            raise ValueError("min_span must be a non-negative integer")
        _unit(self.min_mean_confidence, name="min_mean_confidence")
        _nonnegative_finite(self.min_inward_motion, name="min_inward_motion")
        margin = _unit(self.border_margin_fraction, name="border_margin_fraction")
        if margin >= 0.5:
            raise ValueError("border_margin_fraction must be below 0.5")
        width = _optional_positive(self.image_width, name="image_width")
        height = _optional_positive(self.image_height, name="image_height")
        if (width is None) != (height is None):
            raise ValueError("image_width and image_height must be supplied together")
        if self.require_border_entry and width is None:
            raise ValueError("border-entry filtering requires image dimensions")


@dataclass(frozen=True)
class SequenceBirthFilterSummary:
    sequence: str
    seed_id_count: int
    input_rows: int
    input_ids: int
    birth_tracks: int
    kept_birth_tracks: int
    dropped_birth_tracks: int
    kept_birth_rows: int
    dropped_birth_rows: int
    output_rows: int


@dataclass(frozen=True)
class BirthFilterSummary:
    schema: str
    prediction_path: str
    first_frame_label_dir: str
    output_dir: str
    parameters: BirthFilterParameters
    sequence_count: int
    input_rows: int
    birth_tracks: int
    kept_birth_tracks: int
    dropped_birth_tracks: int
    dropped_birth_rows: int
    output_rows: int
    sequences: tuple[SequenceBirthFilterSummary, ...]


def filter_prediction_set(
    prediction_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    parameters: BirthFilterParameters | None = None,
    sequences: Sequence[str] | None = None,
) -> BirthFilterSummary:
    """Retain seeded tracks and only birth tracks passing truth-free gates."""

    controls = parameters or BirthFilterParameters()
    controls.validate()
    labels = _label_paths(first_frame_label_dir)
    label_map = {path.stem: path for path in labels}
    requested = tuple(dict.fromkeys(str(value) for value in (sequences or ())))
    missing_labels = sorted(set(requested) - set(label_map))
    if missing_labels:
        raise ValueError(f"unknown first-frame sequences: {', '.join(missing_labels)}")

    texts = prediction_texts(prediction_path)
    prediction_map = {Path(name).stem: (name, text) for name, text in texts.items()}
    selected = requested or tuple(sorted(label_map))
    missing_predictions = sorted(set(selected) - set(prediction_map))
    if missing_predictions:
        raise ValueError(
            "prediction input is missing sequences: " + ", ".join(missing_predictions)
        )

    _validate_output_path(prediction_path, first_frame_label_dir, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()

    summaries: list[SequenceBirthFilterSummary] = []
    for sequence in selected:
        prediction_name, text = prediction_map[sequence]
        rows = parse_detection_text(text, source=f"{prediction_path}:{prediction_name}")
        reject_duplicate_keys(rows, label="prediction")
        seed_rows = _seed_rows(label_map[sequence])
        filtered, summary = filter_sequence(
            sequence,
            rows,
            seed_ids={row.object_id for row in seed_rows},
            parameters=controls,
        )
        (output_dir / prediction_name).write_text(
            "".join(format_detection(row) + "\n" for row in filtered),
            encoding="utf-8",
        )
        summaries.append(summary)

    return BirthFilterSummary(
        schema=_SCHEMA,
        prediction_path=str(prediction_path),
        first_frame_label_dir=str(first_frame_label_dir),
        output_dir=str(output_dir),
        parameters=controls,
        sequence_count=len(summaries),
        input_rows=sum(item.input_rows for item in summaries),
        birth_tracks=sum(item.birth_tracks for item in summaries),
        kept_birth_tracks=sum(item.kept_birth_tracks for item in summaries),
        dropped_birth_tracks=sum(item.dropped_birth_tracks for item in summaries),
        dropped_birth_rows=sum(item.dropped_birth_rows for item in summaries),
        output_rows=sum(item.output_rows for item in summaries),
        sequences=tuple(summaries),
    )


def filter_sequence(
    sequence: str,
    rows: Sequence[Detection],
    *,
    seed_ids: set[int],
    parameters: BirthFilterParameters,
) -> tuple[tuple[Detection, ...], SequenceBirthFilterSummary]:
    """Filter one prediction sequence while preserving every seeded identity."""

    parameters.validate()
    grouped: dict[int, list[Detection]] = {}
    for row in rows:
        grouped.setdefault(row.object_id, []).append(row)

    kept: list[Detection] = []
    birth_tracks = 0
    kept_birth_tracks = 0
    dropped_birth_tracks = 0
    kept_birth_rows = 0
    dropped_birth_rows = 0
    for object_id, values in grouped.items():
        track = tuple(sorted(values, key=lambda row: row.frame_id))
        if object_id in seed_ids:
            kept.extend(track)
            continue
        birth_tracks += 1
        if _retain_birth(track, parameters):
            kept.extend(track)
            kept_birth_tracks += 1
            kept_birth_rows += len(track)
        else:
            dropped_birth_tracks += 1
            dropped_birth_rows += len(track)

    kept.sort(key=lambda row: (row.frame_id, row.object_id))
    return tuple(kept), SequenceBirthFilterSummary(
        sequence=sequence,
        seed_id_count=len(seed_ids),
        input_rows=len(rows),
        input_ids=len(grouped),
        birth_tracks=birth_tracks,
        kept_birth_tracks=kept_birth_tracks,
        dropped_birth_tracks=dropped_birth_tracks,
        kept_birth_rows=kept_birth_rows,
        dropped_birth_rows=dropped_birth_rows,
        output_rows=len(kept),
    )


def _retain_birth(
    rows: tuple[Detection, ...],
    parameters: BirthFilterParameters,
) -> bool:
    if not rows or parameters.drop_all_births:
        return False
    span = rows[-1].frame_id - rows[0].frame_id
    mean_confidence = float(np.mean([row.confidence for row in rows]))
    if len(rows) < parameters.min_hits:
        return False
    if span < parameters.min_span:
        return False
    if mean_confidence < parameters.min_mean_confidence:
        return False
    if parameters.require_border_entry and not _is_border_entry(rows, parameters):
        return False
    return True


def _is_border_entry(
    rows: tuple[Detection, ...],
    parameters: BirthFilterParameters,
) -> bool:
    width = parameters.image_width
    height = parameters.image_height
    if width is None or height is None or len(rows) < 2:
        return False
    margin_x = width * parameters.border_margin_fraction
    margin_y = height * parameters.border_margin_fraction
    first = rows[0]
    if not _near_border(first, width, height, margin_x, margin_y):
        return False
    early = rows[: min(len(rows), 4)]
    initial_distance = _border_distance(first, width, height)
    inward_distance = max(
        _border_distance(row, width, height) for row in early[1:]
    ) - initial_distance
    return inward_distance / _scale(first, early[-1]) >= parameters.min_inward_motion


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


def _border_distance(row: Detection, width: float, height: float) -> float:
    return max(
        0.0,
        min(
            row.center_x,
            row.center_y,
            width - row.center_x,
            height - row.center_y,
        ),
    )


def _scale(left: Detection, right: Detection) -> float:
    return max(
        4.0,
        math.sqrt(max(1e-12, left.width * left.height)),
        math.sqrt(max(1e-12, right.width * right.height)),
    )


def _label_paths(label_dir: Path) -> list[Path]:
    if not label_dir.is_dir():
        raise NotADirectoryError(label_dir)
    labels = sorted(label_dir.glob("*.txt"))
    if not labels:
        raise ValueError(f"first-frame label directory contains no .txt files: {label_dir}")
    return labels


def _seed_rows(path: Path) -> tuple[Detection, ...]:
    rows = parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))
    reject_duplicate_keys(rows, label="seed")
    if any(row.frame_id != 1 for row in rows):
        raise ValueError(f"{path}: expected first-frame-only labels")
    return tuple(rows)


def _validate_output_path(
    prediction_path: Path,
    label_dir: Path,
    output_dir: Path,
) -> None:
    output = output_dir.expanduser().resolve()
    labels = label_dir.expanduser().resolve()
    if output == labels or labels in output.parents or output in labels.parents:
        raise ValueError("output directory must be disjoint from first-frame labels")
    if prediction_path.is_dir():
        source = prediction_path.expanduser().resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("output directory must be disjoint from predictions")


def write_summary(summary: BirthFilterSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return parsed


def _positive(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive(value: object, *, name: str) -> float | None:
    return None if value is None else _positive(value, name=name)


def _unit(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--birth-min-hits", type=int, default=5)
    parser.add_argument("--birth-min-span", type=int, default=4)
    parser.add_argument("--birth-min-mean-confidence", type=float, default=0.01)
    parser.add_argument("--birth-require-border-entry", action="store_true")
    parser.add_argument("--birth-min-inward-motion", type=float, default=0.0)
    parser.add_argument("--image-width", type=float)
    parser.add_argument("--image-height", type=float)
    parser.add_argument("--border-margin-fraction", type=float, default=0.08)
    parser.add_argument("--drop-all-births", action="store_true")
    args = parser.parse_args(argv)
    parameters = BirthFilterParameters(
        min_hits=args.birth_min_hits,
        min_span=args.birth_min_span,
        min_mean_confidence=args.birth_min_mean_confidence,
        require_border_entry=args.birth_require_border_entry,
        min_inward_motion=args.birth_min_inward_motion,
        image_width=args.image_width,
        image_height=args.image_height,
        border_margin_fraction=args.border_margin_fraction,
        drop_all_births=args.drop_all_births,
    )
    summary = filter_prediction_set(
        args.prediction_path,
        args.first_frame_label_dir,
        args.output_dir,
        parameters=parameters,
        sequences=args.sequences,
    )
    if args.output_json is not None:
        write_summary(summary, args.output_json)
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
