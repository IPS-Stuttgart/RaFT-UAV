"""Complete short internal LTS trajectory gaps with a two-sided kinematic bridge."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ._records import Detection, format_detection, parse_detection_text, prediction_texts
from .trajectory_box_calibration import BoxCalibrationParameters, _smooth_track

_SUMMARY_SCHEMA = "raft-uav-multi-uav-lts-trajectory-gap-completion-v1"
_EPS = 1e-9


@dataclass(frozen=True)
class GapCompletionParameters:
    """Conservative gates for adding rows only between two observed endpoints."""

    max_gap_frames: int = 2
    max_normalized_speed: float = 5.0
    max_log_size_change: float = 1.0
    min_endpoint_confidence: float = 0.003
    confidence_decay: float = 0.85
    use_smoothed_endpoints: bool = True

    def validate(self) -> None:
        _nonnegative_int(self.max_gap_frames, name="max_gap_frames")
        _positive_finite(self.max_normalized_speed, name="max_normalized_speed")
        _nonnegative_finite(self.max_log_size_change, name="max_log_size_change")
        _unit(self.min_endpoint_confidence, name="min_endpoint_confidence")
        _unit(self.confidence_decay, name="confidence_decay")
        if not isinstance(self.use_smoothed_endpoints, bool):
            raise ValueError("use_smoothed_endpoints must be a Boolean")


@dataclass(frozen=True)
class GapCompletionSequenceSummary:
    sequence: str
    input_rows: int
    output_rows: int
    inserted_rows: int
    eligible_gaps: int
    completed_gaps: int
    rejected_motion_gaps: int
    rejected_size_gaps: int
    rejected_confidence_gaps: int


@dataclass(frozen=True)
class GapCompletionSummary:
    schema: str
    prediction_path: str
    output_dir: str
    parameters: GapCompletionParameters
    sequence_count: int
    input_rows: int
    output_rows: int
    inserted_rows: int
    sequences: tuple[GapCompletionSequenceSummary, ...]


def complete_prediction_set(
    prediction_path: Path,
    output_dir: Path,
    *,
    parameters: GapCompletionParameters | None = None,
    smoother_parameters: BoxCalibrationParameters | None = None,
    sequences: Sequence[str] | None = None,
) -> GapCompletionSummary:
    """Materialize short internal gaps without using ground truth or future labels."""

    config = parameters or GapCompletionParameters()
    config.validate()
    smoother = smoother_parameters or BoxCalibrationParameters()
    smoother.validate()
    texts = prediction_texts(prediction_path)
    requested = tuple(dict.fromkeys(str(value) for value in (sequences or ())))
    available = {Path(name).stem for name in texts}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"unknown prediction sequences: {', '.join(missing)}")
    selected = set(requested)
    _validate_output_path(prediction_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()

    summaries: list[GapCompletionSequenceSummary] = []
    for file_name in sorted(texts):
        sequence = Path(file_name).stem
        if selected and sequence not in selected:
            continue
        rows = tuple(
            parse_detection_text(
                texts[file_name],
                source=f"{prediction_path}:{file_name}",
            )
        )
        completed, summary = complete_sequence(
            sequence,
            rows,
            parameters=config,
            smoother_parameters=smoother,
        )
        (output_dir / file_name).write_text(
            "".join(format_detection(row) + "\n" for row in completed),
            encoding="utf-8",
        )
        summaries.append(summary)

    return GapCompletionSummary(
        schema=_SUMMARY_SCHEMA,
        prediction_path=str(prediction_path),
        output_dir=str(output_dir),
        parameters=config,
        sequence_count=len(summaries),
        input_rows=sum(row.input_rows for row in summaries),
        output_rows=sum(row.output_rows for row in summaries),
        inserted_rows=sum(row.inserted_rows for row in summaries),
        sequences=tuple(summaries),
    )


def complete_sequence(
    sequence: str,
    rows: Sequence[Detection],
    *,
    parameters: GapCompletionParameters,
    smoother_parameters: BoxCalibrationParameters,
) -> tuple[tuple[Detection, ...], GapCompletionSequenceSummary]:
    """Complete eligible gaps for one sequence while preserving all input rows."""

    parameters.validate()
    smoother_parameters.validate()
    by_id: dict[int, list[Detection]] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.frame_id, row.object_id)
        if key in seen:
            raise ValueError(f"{sequence}: duplicate frame/object key {key}")
        seen.add(key)
        by_id.setdefault(row.object_id, []).append(row)

    output = list(rows)
    eligible = 0
    completed_gaps = 0
    rejected_motion = 0
    rejected_size = 0
    rejected_confidence = 0
    inserted = 0
    for object_id in sorted(by_id):
        track = tuple(sorted(by_id[object_id], key=lambda row: row.frame_id))
        smoothed = (
            _smooth_track(track, smoother_parameters)
            if parameters.use_smoothed_endpoints and len(track) >= 2
            else ()
        )
        for index, (left, right) in enumerate(zip(track, track[1:])):
            gap = right.frame_id - left.frame_id - 1
            if gap <= 0 or gap > parameters.max_gap_frames:
                continue
            eligible += 1
            if min(left.confidence, right.confidence) < parameters.min_endpoint_confidence:
                rejected_confidence += 1
                continue
            duration = right.frame_id - left.frame_id
            scale = max(
                1.0,
                0.5 * (math.sqrt(left.width * left.height) + math.sqrt(right.width * right.height)),
            )
            normalized_speed = math.hypot(
                right.center_x - left.center_x,
                right.center_y - left.center_y,
            ) / (scale * duration)
            if normalized_speed > parameters.max_normalized_speed:
                rejected_motion += 1
                continue
            size_change = abs(math.log(right.width / left.width)) + abs(
                math.log(right.height / left.height)
            )
            if size_change > parameters.max_log_size_change:
                rejected_size += 1
                continue
            if left.class_id != right.class_id:
                rejected_size += 1
                continue

            left_state = None
            right_state = None
            if smoothed:
                left_state = smoothed[index].state
                right_state = smoothed[index + 1].state
            new_rows = _bridge_gap(
                left,
                right,
                gap,
                left_state=left_state,
                right_state=right_state,
                confidence_decay=parameters.confidence_decay,
            )
            output.extend(new_rows)
            inserted += len(new_rows)
            completed_gaps += 1

    output.sort(key=lambda row: (row.frame_id, row.object_id))
    if len({(row.frame_id, row.object_id) for row in output}) != len(output):
        raise ValueError(f"{sequence}: gap completion created duplicate frame/object keys")
    return tuple(output), GapCompletionSequenceSummary(
        sequence=sequence,
        input_rows=len(rows),
        output_rows=len(output),
        inserted_rows=inserted,
        eligible_gaps=eligible,
        completed_gaps=completed_gaps,
        rejected_motion_gaps=rejected_motion,
        rejected_size_gaps=rejected_size,
        rejected_confidence_gaps=rejected_confidence,
    )


def _bridge_gap(
    left: Detection,
    right: Detection,
    gap: int,
    *,
    left_state,
    right_state,
    confidence_decay: float,
) -> tuple[Detection, ...]:
    duration = float(right.frame_id - left.frame_id)
    if left_state is None:
        x0, y0 = left.center_x, left.center_y
        x1, y1 = right.center_x, right.center_y
        vx0 = vx1 = (x1 - x0) / duration
        vy0 = vy1 = (y1 - y0) / duration
        log_w0, log_h0 = math.log(left.width), math.log(left.height)
        log_w1, log_h1 = math.log(right.width), math.log(right.height)
    else:
        x0, y0, vx0, vy0, log_w0, log_h0 = (float(value) for value in left_state)
        x1, y1, vx1, vy1, log_w1, log_h1 = (float(value) for value in right_state)

    rows: list[Detection] = []
    for offset in range(1, gap + 1):
        u = offset / duration
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        center_x = h00 * x0 + h10 * duration * vx0 + h01 * x1 + h11 * duration * vx1
        center_y = h00 * y0 + h10 * duration * vy0 + h01 * y1 + h11 * duration * vy1
        log_width = (1.0 - u) * log_w0 + u * log_w1
        log_height = (1.0 - u) * log_h0 + u * log_h1
        width = max(_EPS, math.exp(log_width))
        height = max(_EPS, math.exp(log_height))
        endpoint_distance = min(offset, gap + 1 - offset)
        confidence = min(left.confidence, right.confidence) * confidence_decay**endpoint_distance
        rows.append(
            Detection(
                left.frame_id + offset,
                left.object_id,
                center_x - 0.5 * width,
                center_y - 0.5 * height,
                width,
                height,
                max(0.0, min(1.0, confidence)),
                left.class_id,
                min(left.visibility, right.visibility),
            )
        )
    return tuple(rows)


def _validate_output_path(prediction_path: Path, output_dir: Path) -> None:
    if prediction_path.is_dir():
        source = prediction_path.expanduser().resolve()
        output = output_dir.expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("output directory must not alias or be nested in predictions")


def write_summary(summary: GapCompletionSummary, path: Path) -> None:
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


def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _unit(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--max-gap-frames", type=int, default=2)
    parser.add_argument("--max-normalized-speed", type=float, default=5.0)
    parser.add_argument("--max-log-size-change", type=float, default=1.0)
    parser.add_argument("--min-endpoint-confidence", type=float, default=0.003)
    parser.add_argument("--confidence-decay", type=float, default=0.85)
    parser.add_argument("--raw-endpoints", action="store_true")
    parser.add_argument("--process-accel-sigma", type=float, default=0.50)
    parser.add_argument("--process-log-size-sigma", type=float, default=0.035)
    parser.add_argument("--center-measurement-sigma", type=float, default=0.30)
    parser.add_argument("--log-size-measurement-sigma", type=float, default=0.12)
    args = parser.parse_args(argv)
    parameters = GapCompletionParameters(
        max_gap_frames=args.max_gap_frames,
        max_normalized_speed=args.max_normalized_speed,
        max_log_size_change=args.max_log_size_change,
        min_endpoint_confidence=args.min_endpoint_confidence,
        confidence_decay=args.confidence_decay,
        use_smoothed_endpoints=not args.raw_endpoints,
    )
    smoother = BoxCalibrationParameters(
        process_accel_sigma=args.process_accel_sigma,
        process_log_size_sigma=args.process_log_size_sigma,
        center_measurement_sigma=args.center_measurement_sigma,
        log_size_measurement_sigma=args.log_size_measurement_sigma,
    )
    summary = complete_prediction_set(
        args.prediction_path,
        args.output_dir,
        parameters=parameters,
        smoother_parameters=smoother,
        sequences=args.sequences,
    )
    if args.output_json is not None:
        write_summary(summary, args.output_json)
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
