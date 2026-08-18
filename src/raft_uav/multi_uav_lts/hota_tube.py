"""Conservative short-gap tubes for the Multi-UAV LTS Codabench objective."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from ._records import (
    Detection,
    box_iou,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
)


@dataclass(frozen=True)
class HotaTubeParameters:
    max_gap: int
    base_inflation: float
    velocity_inflation: float
    max_scale: float
    conflict_iou: float
    confidence_decay: float
    min_track_observations: int


@dataclass(frozen=True)
class SequenceHotaTubeSummary:
    sequence: str
    input_rows: int
    output_rows: int
    eligible_gaps: int
    proposed_rows: int
    inserted_rows: int
    conflict_rejected_rows: int
    maximum_scale: float


@dataclass(frozen=True)
class HotaTubeSummary:
    schema: str
    prediction_path: str
    output_dir: str
    parameters: HotaTubeParameters
    sequence_count: int
    input_rows: int
    output_rows: int
    eligible_gaps: int
    proposed_rows: int
    inserted_rows: int
    conflict_rejected_rows: int
    maximum_scale: float
    sequences: tuple[SequenceHotaTubeSummary, ...]


@dataclass(frozen=True)
class _ProposedRow:
    row: Detection
    scale: float


def apply_hota_tube(
    prediction_path: Path,
    output_dir: Path,
    *,
    max_gap: int = 1,
    base_inflation: float = 0.75,
    velocity_inflation: float = 0.5,
    max_scale: float = 3.0,
    conflict_iou: float = 0.3,
    confidence_decay: float = 0.9,
    min_track_observations: int = 2,
    sequences: Iterable[str] | None = None,
) -> HotaTubeSummary:
    """Interpolate short internal gaps and widen only the synthetic rows."""
    parameters = _parameters(
        max_gap=max_gap,
        base_inflation=base_inflation,
        velocity_inflation=velocity_inflation,
        max_scale=max_scale,
        conflict_iou=conflict_iou,
        confidence_decay=confidence_decay,
        min_track_observations=min_track_observations,
    )
    prediction_path = prediction_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _guard_paths(prediction_path, output_dir)
    texts = prediction_texts(prediction_path)
    available = {Path(name).stem for name in texts}
    requested = set(sequences or ())
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"unknown prediction sequences: {', '.join(missing)}")

    prepared: dict[str, tuple[Detection, ...]] = {}
    summaries: list[SequenceHotaTubeSummary] = []
    for name, text in sorted(texts.items()):
        sequence = Path(name).stem
        if requested and sequence not in requested:
            continue
        rows = parse_detection_text(text, source=f"{prediction_path}:{name}")
        reject_duplicate_keys(rows, label="predictions")
        for row in rows:
            if not -1.0 <= row.confidence <= 1.0:
                raise ValueError(
                    f"{prediction_path}:{name}: confidence must be in [-1, 1]"
                )
        output_rows, summary = _tube_sequence(sequence, tuple(rows), parameters)
        prepared[name] = output_rows
        summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()
    for name, rows in prepared.items():
        (output_dir / name).write_text(
            "".join(format_detection(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    return _summary(prediction_path, output_dir, parameters, tuple(summaries))


def _parameters(**raw: object) -> HotaTubeParameters:
    max_gap = _nonnegative_int(raw["max_gap"], name="max_gap")
    min_track = _nonnegative_int(
        raw["min_track_observations"],
        name="min_track_observations",
    )
    if min_track < 2:
        raise ValueError("min_track_observations must be at least 2")
    base = _nonnegative_float(raw["base_inflation"], name="base_inflation")
    velocity = _nonnegative_float(
        raw["velocity_inflation"],
        name="velocity_inflation",
    )
    maximum = _nonnegative_float(raw["max_scale"], name="max_scale")
    if maximum < 1.0:
        raise ValueError("max_scale must be at least 1")
    return HotaTubeParameters(
        max_gap=max_gap,
        base_inflation=base,
        velocity_inflation=velocity,
        max_scale=maximum,
        conflict_iou=_unit_float(raw["conflict_iou"], name="conflict_iou"),
        confidence_decay=_unit_float(
            raw["confidence_decay"],
            name="confidence_decay",
        ),
        min_track_observations=min_track,
    )


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        parsed_float = float(value)
        parsed = int(parsed_float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if not math.isfinite(parsed_float) or parsed_float != parsed or parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return parsed


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return parsed


def _unit_float(value: object, *, name: str) -> float:
    parsed = _nonnegative_float(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _guard_paths(prediction_path: Path, output_dir: Path) -> None:
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if prediction_path.is_dir() and (
        output_dir == prediction_path or prediction_path in output_dir.parents
    ):
        raise ValueError("output directory must not alias or be nested in predictions")


def _tube_sequence(
    sequence: str,
    rows: tuple[Detection, ...],
    parameters: HotaTubeParameters,
) -> tuple[tuple[Detection, ...], SequenceHotaTubeSummary]:
    tracks: dict[int, list[Detection]] = {}
    observed_by_frame: dict[int, list[Detection]] = {}
    for row in rows:
        tracks.setdefault(row.object_id, []).append(row)
        observed_by_frame.setdefault(row.frame_id, []).append(row)
    proposed: list[_ProposedRow] = []
    eligible_gaps = 0
    for track_rows in tracks.values():
        ordered = sorted(track_rows, key=lambda row: row.frame_id)
        if len(ordered) < parameters.min_track_observations:
            continue
        for index, (left, right) in enumerate(zip(ordered, ordered[1:], strict=False)):
            missing = right.frame_id - left.frame_id - 1
            if missing <= 0 or missing > parameters.max_gap:
                continue
            if left.class_id != right.class_id:
                continue
            eligible_gaps += 1
            previous = ordered[index - 1] if index > 0 else None
            following_index = index + 2
            following = (
                ordered[following_index]
                if following_index < len(ordered)
                else None
            )
            for step in range(1, missing + 1):
                row, scale = _interpolated_row(
                    previous,
                    left,
                    right,
                    following,
                    step=step,
                    parameters=parameters,
                )
                proposed.append(_ProposedRow(row, scale))

    accepted: list[_ProposedRow] = []
    rejected = 0
    frame_rows = {
        frame: list(items) for frame, items in observed_by_frame.items()
    }
    for proposal in sorted(
        proposed,
        key=lambda item: (
            item.row.frame_id,
            -item.row.confidence,
            item.row.object_id,
        ),
    ):
        competitors = frame_rows.setdefault(proposal.row.frame_id, [])
        if any(
            row.object_id != proposal.row.object_id
            and box_iou(row, proposal.row) >= parameters.conflict_iou
            for row in competitors
        ):
            rejected += 1
            continue
        competitors.append(proposal.row)
        accepted.append(proposal)

    output = tuple(
        sorted(
            (*rows, *(item.row for item in accepted)),
            key=lambda row: (row.frame_id, row.object_id),
        )
    )
    maximum_scale = max((item.scale for item in accepted), default=1.0)
    return output, SequenceHotaTubeSummary(
        sequence=sequence,
        input_rows=len(rows),
        output_rows=len(output),
        eligible_gaps=eligible_gaps,
        proposed_rows=len(proposed),
        inserted_rows=len(accepted),
        conflict_rejected_rows=rejected,
        maximum_scale=maximum_scale,
    )


def _interpolated_row(
    previous: Detection | None,
    left: Detection,
    right: Detection,
    following: Detection | None,
    *,
    step: int,
    parameters: HotaTubeParameters,
) -> tuple[Detection, float]:
    interval = right.frame_id - left.frame_id
    fraction = step / interval
    center_x = _lerp(left.center_x, right.center_x, fraction)
    center_y = _lerp(left.center_y, right.center_y, fraction)
    width = math.exp(
        _lerp(math.log(left.width), math.log(right.width), fraction)
    )
    height = math.exp(
        _lerp(math.log(left.height), math.log(right.height), fraction)
    )
    relative_uncertainty = _relative_motion_uncertainty(
        previous,
        left,
        right,
        following,
        step=step,
        center_x=center_x,
        center_y=center_y,
        scale=max(4.0, math.sqrt(width * height)),
    )
    phase = 4.0 * fraction * (1.0 - fraction)
    scale_factor = min(
        parameters.max_scale,
        1.0
        + phase
        * (
            parameters.base_inflation
            + parameters.velocity_inflation * relative_uncertainty
        ),
    )
    scaled_width = width * scale_factor
    scaled_height = height * scale_factor
    confidence = _interpolated_confidence(
        left.confidence,
        right.confidence,
        parameters.confidence_decay,
        step=step,
        interval=interval,
    )
    return (
        replace(
            left,
            frame_id=left.frame_id + step,
            x1=center_x - 0.5 * scaled_width,
            y1=center_y - 0.5 * scaled_height,
            width=scaled_width,
            height=scaled_height,
            confidence=confidence,
            visibility=min(left.visibility, right.visibility),
        ),
        scale_factor,
    )


def _relative_motion_uncertainty(
    previous: Detection | None,
    left: Detection,
    right: Detection,
    following: Detection | None,
    *,
    step: int,
    center_x: float,
    center_y: float,
    scale: float,
) -> float:
    interval = right.frame_id - left.frame_id
    distances: list[float] = []
    if previous is not None:
        delta = left.frame_id - previous.frame_id
        if delta > 0:
            vx = (left.center_x - previous.center_x) / delta
            vy = (left.center_y - previous.center_y) / delta
            distances.append(
                math.hypot(
                    left.center_x + vx * step - center_x,
                    left.center_y + vy * step - center_y,
                )
            )
    if following is not None:
        delta = following.frame_id - right.frame_id
        if delta > 0:
            vx = (following.center_x - right.center_x) / delta
            vy = (following.center_y - right.center_y) / delta
            remaining = interval - step
            distances.append(
                math.hypot(
                    right.center_x - vx * remaining - center_x,
                    right.center_y - vy * remaining - center_y,
                )
            )
    return max(distances, default=0.0) / scale


def _interpolated_confidence(
    left: float,
    right: float,
    decay: float,
    *,
    step: int,
    interval: int,
) -> float:
    if left < 0.0 or right < 0.0:
        return -1.0
    distance = min(step, interval - step)
    return min(left, right) * decay**distance


def _lerp(left: float, right: float, fraction: float) -> float:
    return left + fraction * (right - left)


def _summary(
    prediction_path: Path,
    output_dir: Path,
    parameters: HotaTubeParameters,
    rows: tuple[SequenceHotaTubeSummary, ...],
) -> HotaTubeSummary:
    def total(name: str) -> int:
        return sum(int(getattr(row, name)) for row in rows)

    return HotaTubeSummary(
        schema="raft-uav-multi-uav-lts-hota-tube-v1",
        prediction_path=str(prediction_path),
        output_dir=str(output_dir),
        parameters=parameters,
        sequence_count=len(rows),
        input_rows=total("input_rows"),
        output_rows=total("output_rows"),
        eligible_gaps=total("eligible_gaps"),
        proposed_rows=total("proposed_rows"),
        inserted_rows=total("inserted_rows"),
        conflict_rejected_rows=total("conflict_rejected_rows"),
        maximum_scale=max(
            (row.maximum_scale for row in rows),
            default=1.0,
        ),
        sequences=rows,
    )


def write_summary(summary: HotaTubeSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--base-inflation", type=float, default=0.75)
    parser.add_argument("--velocity-inflation", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=3.0)
    parser.add_argument("--conflict-iou", type=float, default=0.3)
    parser.add_argument("--confidence-decay", type=float, default=0.9)
    parser.add_argument("--min-track-observations", type=int, default=2)
    args = parser.parse_args(argv)
    keywords = vars(args).copy()
    prediction_path = keywords.pop("prediction_path")
    output_dir = keywords.pop("output_dir")
    output_json = keywords.pop("output_json")
    summary = apply_hota_tube(prediction_path, output_dir, **keywords)
    if output_json is not None:
        write_summary(summary, output_json)
    print(f"sequence_count={summary.sequence_count}")
    print(f"inserted_rows={summary.inserted_rows}")
    print(f"conflict_rejected_rows={summary.conflict_rejected_rows}")
    print(f"maximum_scale={summary.maximum_scale:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
