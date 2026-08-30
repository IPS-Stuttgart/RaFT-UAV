"""HOTA(0)-aware uncertainty box expansion and guarded short-gap repair."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ._full_stack_io import (
    clip_box,
    group_id,
    image_index,
    load_gray,
    prediction_files,
    read_rows,
    write_json,
    write_rows,
)
from ._records import Detection


@dataclass(frozen=True)
class MetricBoxConfig:
    base_scale: float = 1.0
    tiny_side: float = 18.0
    tiny_gain: float = 0.18
    innovation_gain: float = 0.10
    gap_gain: float = 0.05
    maximum_scale: float = 1.35
    max_interpolation_gap: int = 0
    interpolation_confidence_decay: float = 0.75
    interpolation_visibility: float = 0.5


def _local_uncertainty(track: list[Detection], index: int) -> tuple[float, int]:
    row = track[index]
    side = max(math.sqrt(row.width * row.height), 2.0)
    residual = 0.0
    surrounding_gap = 0
    if 0 < index < len(track) - 1:
        previous = track[index - 1]
        following = track[index + 1]
        denominator = following.frame_id - previous.frame_id
        if denominator > 0:
            ratio = (row.frame_id - previous.frame_id) / denominator
            predicted_x = previous.center_x + ratio * (following.center_x - previous.center_x)
            predicted_y = previous.center_y + ratio * (following.center_y - previous.center_y)
            residual = math.hypot(row.center_x - predicted_x, row.center_y - predicted_y) / side
        surrounding_gap = max(
            row.frame_id - previous.frame_id - 1,
            following.frame_id - row.frame_id - 1,
        )
    elif index >= 2:
        first = track[index - 2]
        previous = track[index - 1]
        denominator = previous.frame_id - first.frame_id
        if denominator > 0:
            ratio = (row.frame_id - previous.frame_id) / denominator
            predicted_x = previous.center_x + ratio * (previous.center_x - first.center_x)
            predicted_y = previous.center_y + ratio * (previous.center_y - first.center_y)
            residual = math.hypot(row.center_x - predicted_x, row.center_y - predicted_y) / side
        surrounding_gap = row.frame_id - previous.frame_id - 1
    elif index + 2 < len(track):
        following = track[index + 1]
        second = track[index + 2]
        denominator = second.frame_id - following.frame_id
        if denominator > 0:
            ratio = (following.frame_id - row.frame_id) / denominator
            predicted_x = following.center_x - ratio * (second.center_x - following.center_x)
            predicted_y = following.center_y - ratio * (second.center_y - following.center_y)
            residual = math.hypot(row.center_x - predicted_x, row.center_y - predicted_y) / side
        surrounding_gap = following.frame_id - row.frame_id - 1
    return float(np.clip(residual, 0.0, 8.0)), max(0, surrounding_gap)


def expansion_scale(
    row: Detection,
    uncertainty: float,
    surrounding_gap: int,
    config: MetricBoxConfig,
) -> float:
    side = math.sqrt(max(row.width * row.height, 1e-6))
    tiny_fraction = float(np.clip((config.tiny_side - side) / max(config.tiny_side, 1e-6), 0.0, 1.0))
    scale = (
        config.base_scale
        + config.tiny_gain * tiny_fraction
        + config.innovation_gain * math.sqrt(max(uncertainty, 0.0))
        + config.gap_gain * math.sqrt(max(surrounding_gap, 0))
    )
    return float(np.clip(scale, 1.0, config.maximum_scale))


def expand_track(
    rows: list[Detection],
    config: MetricBoxConfig = MetricBoxConfig(),
    image_shape: tuple[int, int] | None = None,
) -> tuple[list[Detection], dict]:
    track = sorted(rows, key=lambda row: row.frame_id)
    expanded = []
    scales = []
    for index, row in enumerate(track):
        uncertainty, surrounding_gap = _local_uncertainty(track, index)
        scale = expansion_scale(row, uncertainty, surrounding_gap, config)
        scales.append(scale)
        width = row.width * scale
        height = row.height * scale
        candidate = replace(
            row,
            x1=row.center_x - 0.5 * width,
            y1=row.center_y - 0.5 * height,
            width=width,
            height=height,
        )
        if image_shape is not None:
            candidate = clip_box(candidate, image_shape[1], image_shape[0])
        expanded.append(candidate)
    return expanded, {
        "mean_scale": float(np.mean(scales)) if scales else 1.0,
        "max_scale": max(scales, default=1.0),
    }


def interpolate_short_gaps(
    rows: list[Detection],
    config: MetricBoxConfig,
    image_shape: tuple[int, int] | None = None,
) -> tuple[list[Detection], int]:
    if config.max_interpolation_gap <= 0:
        return sorted(rows, key=lambda row: row.frame_id), 0
    track = sorted(rows, key=lambda row: row.frame_id)
    output: list[Detection] = []
    created = 0
    for index, left in enumerate(track):
        output.append(left)
        if index + 1 >= len(track):
            continue
        right = track[index + 1]
        missing = right.frame_id - left.frame_id - 1
        if missing <= 0 or missing > config.max_interpolation_gap:
            continue
        denominator = right.frame_id - left.frame_id
        for offset in range(1, missing + 1):
            ratio = offset / denominator
            center_x = left.center_x + ratio * (right.center_x - left.center_x)
            center_y = left.center_y + ratio * (right.center_y - left.center_y)
            log_width = (1.0 - ratio) * math.log(left.width) + ratio * math.log(right.width)
            log_height = (1.0 - ratio) * math.log(left.height) + ratio * math.log(right.height)
            width = math.exp(log_width)
            height = math.exp(log_height)
            confidence = (
                min(left.confidence, right.confidence)
                * config.interpolation_confidence_decay ** min(offset, denominator - offset)
            )
            candidate = Detection(
                left.frame_id + offset,
                left.object_id,
                center_x - 0.5 * width,
                center_y - 0.5 * height,
                width,
                height,
                confidence,
                left.class_id,
                min(left.visibility, right.visibility, config.interpolation_visibility),
            )
            if image_shape is not None:
                candidate = clip_box(candidate, image_shape[1], image_shape[0])
            output.append(candidate)
            created += 1
    output.sort(key=lambda row: row.frame_id)
    return output, created


def transform_sequence(
    rows: list[Detection],
    config: MetricBoxConfig = MetricBoxConfig(),
    image_shape: tuple[int, int] | None = None,
) -> tuple[list[Detection], dict]:
    output = []
    track_diagnostics = {}
    interpolation_count = 0
    for identity, track in group_id(rows).items():
        repaired, created = interpolate_short_gaps(track, config, image_shape)
        interpolation_count += created
        expanded, diagnostics = expand_track(repaired, config, image_shape)
        output.extend(expanded)
        track_diagnostics[str(identity)] = diagnostics
    return output, {
        "track_count": len(track_diagnostics),
        "interpolation_count": interpolation_count,
        "mean_scale": float(
            np.mean([value["mean_scale"] for value in track_diagnostics.values()])
        )
        if track_diagnostics
        else 1.0,
        "maximum_scale": max(
            (value["max_scale"] for value in track_diagnostics.values()),
            default=1.0,
        ),
    }


def transform_directory(
    input_dir: Path,
    output_dir: Path,
    sequence_names: list[str] | None = None,
    config: MetricBoxConfig = MetricBoxConfig(),
    image_root: Path | None = None,
) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for file in prediction_files(input_dir):
        if allowed is not None and file.stem not in allowed:
            continue
        image_shape = None
        if image_root is not None:
            paths = image_index(image_root / file.stem)
            if paths:
                image_shape = load_gray(paths[min(paths)]).shape
        transformed, diagnostics = transform_sequence(
            read_rows(file),
            config,
            image_shape,
        )
        write_rows(output_dir / file.name, transformed)
        summary[file.stem] = diagnostics
    return {"sequences": summary, "config": config.__dict__}


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--tiny-gain", type=float, default=0.18)
    parser.add_argument("--innovation-gain", type=float, default=0.10)
    parser.add_argument("--maximum-scale", type=float, default=1.35)
    parser.add_argument("--max-interpolation-gap", type=int, default=0)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = transform_directory(
        args.input_dir,
        args.output_dir,
        _names(args.sequence_list),
        MetricBoxConfig(
            tiny_gain=args.tiny_gain,
            innovation_gain=args.innovation_gain,
            maximum_scale=args.maximum_scale,
            max_interpolation_gap=args.max_interpolation_gap,
        ),
        args.image_root,
    )
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
