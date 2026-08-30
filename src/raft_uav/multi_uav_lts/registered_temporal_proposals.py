"""Symmetric, camera-registered temporal residual proposals for tiny UAVs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ._full_stack_io import (
    crop_box,
    group_id,
    image_index,
    load_gray,
    prediction_files,
    read_rows,
    write_json,
    write_rows,
)
from ._records import Detection
from .scene_stabilization import StabilizationConfig, phase_translation, translate_image
from .track_conditioned_proposals import RoiConfig, predict_roi


@dataclass(frozen=True)
class RegisteredTemporalConfig:
    offsets: tuple[int, ...] = (-4, -2, -1, 1, 2, 4)
    robust_z: float = 3.2
    min_area: int = 2
    max_area: int = 625
    centre_gate: float = 0.75
    min_neighbours: int = 2
    opening_iterations: int = 0
    registration: StabilizationConfig = StabilizationConfig(
        max_shift=24.0,
        min_peak_ratio=1.08,
        mask_scale=1.0,
        downsample=1,
    )
    roi: RoiConfig = RoiConfig(box_scale=4.5, upscale=1.0)


def registered_residual_components(
    current: np.ndarray,
    neighbours: list[np.ndarray],
    config: RegisteredTemporalConfig = RegisteredTemporalConfig(),
) -> tuple[list[tuple[int, int, int, int, float]], dict]:
    """Return connected residual components after quality-gated registration."""

    aligned: list[np.ndarray] = []
    estimates = []
    for neighbour in neighbours:
        height = min(current.shape[0], neighbour.shape[0])
        width = min(current.shape[1], neighbour.shape[1])
        reference = current[:height, :width]
        moving = neighbour[:height, :width]
        estimate = phase_translation(reference, moving, config.registration)
        estimates.append(estimate)
        if estimate.accepted:
            aligned.append(translate_image(moving, estimate.dy, estimate.dx))
    diagnostics = {
        "neighbour_count": len(neighbours),
        "accepted_registrations": len(aligned),
        "peak_ratios": [estimate.peak_ratio for estimate in estimates],
    }
    if len(aligned) < config.min_neighbours:
        return [], diagnostics

    height = min([current.shape[0], *[image.shape[0] for image in aligned]])
    width = min([current.shape[1], *[image.shape[1] for image in aligned]])
    centre = current[:height, :width]
    stack = np.stack([image[:height, :width] for image in aligned])
    background = np.median(stack, axis=0)
    temporal_mad = np.median(np.abs(stack - background), axis=0)
    residual = np.abs(centre - background)
    noise_floor = float(np.median(residual))
    noise_mad = float(np.median(np.abs(residual - noise_floor))) + 1e-6
    adaptive_noise = noise_floor + config.robust_z * 1.4826 * (
        noise_mad + float(np.median(temporal_mad))
    )
    mask = residual > adaptive_noise
    if config.opening_iterations > 0:
        mask = ndimage.binary_opening(mask, iterations=config.opening_iterations)
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    components: list[tuple[int, int, int, int, float]] = []
    for label_id, slices in enumerate(objects, 1):
        if slices is None:
            continue
        selected = labels[slices] == label_id
        area = int(np.sum(selected))
        if not config.min_area <= area <= config.max_area:
            continue
        y_slice, x_slice = slices
        values = residual[slices][selected]
        score = float(np.mean(values) / max(adaptive_noise, 1e-6))
        stability = float(np.mean(temporal_mad[slices][selected]))
        score /= 1.0 + stability / max(noise_mad, 1e-6)
        components.append(
            (
                x_slice.start,
                y_slice.start,
                x_slice.stop - x_slice.start,
                y_slice.stop - y_slice.start,
                float(np.clip(score / 3.0, 0.0, 1.0)),
            )
        )
    diagnostics["label_count"] = int(count)
    diagnostics["component_count"] = len(components)
    return components, diagnostics


def generate_registered_temporal(
    image_root: Path,
    tracks_dir: Path,
    output_dir: Path,
    sequence_names: list[str] | None = None,
    config: RegisteredTemporalConfig = RegisteredTemporalConfig(),
) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary: dict[str, dict] = {}
    for track_file in prediction_files(tracks_dir):
        name = track_file.stem
        if allowed is not None and name not in allowed:
            continue
        frame_paths = image_index(image_root / name)
        cache: dict[int, np.ndarray] = {}

        def image(frame: int) -> np.ndarray:
            if frame not in cache:
                cache[frame] = load_gray(frame_paths[frame])
            return cache[frame]

        tracks = group_id(read_rows(track_file))
        proposals: list[Detection] = []
        accepted_registrations = 0
        attempted_registrations = 0
        for frame in sorted(frame_paths):
            current_image = image(frame)
            local: list[tuple[float, float, float, float, float]] = []
            for history in tracks.values():
                try:
                    x, y, width, height = predict_roi(
                        history,
                        frame,
                        current_image.shape,
                        config.roi,
                    )
                except ValueError:
                    continue
                current_crop, bounds = crop_box(
                    current_image,
                    x,
                    y,
                    x + width,
                    y + height,
                )
                neighbour_crops: list[np.ndarray] = []
                for offset in config.offsets:
                    other = frame + offset
                    if other not in frame_paths:
                        continue
                    crop, _ = crop_box(image(other), x, y, x + width, y + height)
                    neighbour_crops.append(crop)
                components, diagnostics = registered_residual_components(
                    current_crop,
                    neighbour_crops,
                    config,
                )
                accepted_registrations += diagnostics["accepted_registrations"]
                attempted_registrations += diagnostics["neighbour_count"]
                crop_height, crop_width = current_crop.shape
                for bx, by, bw, bh, score in components:
                    normalized_distance = np.hypot(
                        (bx + 0.5 * bw - 0.5 * crop_width) / max(crop_width, 1),
                        (by + 0.5 * bh - 0.5 * crop_height) / max(crop_height, 1),
                    )
                    if normalized_distance <= config.centre_gate:
                        local.append(
                            (
                                bounds[0] + bx,
                                bounds[1] + by,
                                float(max(bw, 1)),
                                float(max(bh, 1)),
                                score,
                            )
                        )
            seen: set[tuple[float, float, float, float]] = set()
            for x, y, width, height, score in sorted(local, key=lambda row: -row[4]):
                key = tuple(round(value, 1) for value in (x, y, width, height))
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(
                    Detection(
                        frame,
                        len(proposals) + 1,
                        x,
                        y,
                        width,
                        height,
                        score,
                        1,
                        1.0,
                    )
                )
        write_rows(output_dir / f"{name}.txt", proposals)
        summary[name] = {
            "proposal_count": len(proposals),
            "accepted_registrations": accepted_registrations,
            "attempted_registrations": attempted_registrations,
        }
    return {
        "sequences": summary,
        "total_proposals": sum(value["proposal_count"] for value in summary.values()),
        "config": {
            "offsets": list(config.offsets),
            "robust_z": config.robust_z,
            "centre_gate": config.centre_gate,
        },
    }


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--robust-z", type=float, default=3.2)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_registered_temporal(
        args.image_root,
        args.tracks_dir,
        args.output_dir,
        _names(args.sequence_list),
        RegisteredTemporalConfig(robust_z=args.robust_z),
    )
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
