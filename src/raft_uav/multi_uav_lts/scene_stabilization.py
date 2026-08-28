"""Background-only camera translation estimation for Multi-UAV LTS sequences.

The estimator deliberately uses a small, deterministic phase-correlation model.
It masks known target boxes, rejects weak or implausible registrations, and
returns a cumulative translation that can be injected into association geometry
without changing submitted image-coordinate boxes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import ndimage

from ._full_stack_io import group_frame, image_index, load_gray, prediction_files, read_rows, write_json
from ._records import Detection


@dataclass(frozen=True)
class StabilizationConfig:
    max_shift: float = 48.0
    min_peak_ratio: float = 1.18
    mask_scale: float = 3.0
    min_texture_std: float = 0.01
    downsample: int = 2
    subpixel: bool = True


@dataclass(frozen=True)
class TranslationEstimate:
    dy: float
    dx: float
    peak_ratio: float
    accepted: bool
    reason: str


def _crop_common(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    if height < 4 or width < 4:
        raise ValueError("phase correlation requires images of at least 4x4 pixels")
    return left[:height, :width], right[:height, :width]


def mask_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    *,
    scale: float = 3.0,
) -> np.ndarray:
    """Replace target neighborhoods by the image median before registration."""

    masked = np.asarray(image, dtype=np.float32).copy()
    height, width = masked.shape[:2]
    fill = float(np.median(masked))
    for row in detections:
        box_width = max(row.width * scale, row.width + 4.0)
        box_height = max(row.height * scale, row.height + 4.0)
        x1 = max(0, int(np.floor(row.center_x - 0.5 * box_width)))
        y1 = max(0, int(np.floor(row.center_y - 0.5 * box_height)))
        x2 = min(width, int(np.ceil(row.center_x + 0.5 * box_width)))
        y2 = min(height, int(np.ceil(row.center_y + 0.5 * box_height)))
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = fill
    return masked


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    size = len(values)
    left = float(values[(index - 1) % size])
    center = float(values[index])
    right = float(values[(index + 1) % size])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def phase_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    config: StabilizationConfig = StabilizationConfig(),
) -> TranslationEstimate:
    """Estimate the shift applied to ``moving`` so it aligns with ``reference``."""

    ref, mov = _crop_common(np.asarray(reference, float), np.asarray(moving, float))
    stride = max(1, int(config.downsample))
    ref = ref[::stride, ::stride]
    mov = mov[::stride, ::stride]
    ref = ref - float(np.mean(ref))
    mov = mov - float(np.mean(mov))
    ref_std = float(np.std(ref))
    mov_std = float(np.std(mov))
    if min(ref_std, mov_std) < config.min_texture_std:
        return TranslationEstimate(0.0, 0.0, 0.0, False, "insufficient_texture")

    window = np.outer(np.hanning(ref.shape[0]), np.hanning(ref.shape[1]))
    ref *= window
    mov *= window
    cross = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    magnitude = np.abs(cross)
    cross /= np.maximum(magnitude, 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak = float(correlation[peak_y, peak_x])

    excluded = correlation.copy()
    for y_offset in range(-2, 3):
        for x_offset in range(-2, 3):
            excluded[(peak_y + y_offset) % excluded.shape[0], (peak_x + x_offset) % excluded.shape[1]] = -np.inf
    second = float(np.max(excluded))
    baseline = float(np.median(np.abs(correlation))) + 1e-12
    peak_ratio = (peak - baseline) / max(second - baseline, baseline)

    dy = float(peak_y if peak_y <= correlation.shape[0] // 2 else peak_y - correlation.shape[0])
    dx = float(peak_x if peak_x <= correlation.shape[1] // 2 else peak_x - correlation.shape[1])
    if config.subpixel:
        dy += _parabolic_offset(correlation[:, peak_x], peak_y)
        dx += _parabolic_offset(correlation[peak_y, :], peak_x)
    dy *= stride
    dx *= stride

    if np.hypot(dy, dx) > config.max_shift:
        return TranslationEstimate(0.0, 0.0, peak_ratio, False, "shift_gate")
    if not np.isfinite(peak_ratio) or peak_ratio < config.min_peak_ratio:
        return TranslationEstimate(0.0, 0.0, peak_ratio, False, "weak_peak")
    return TranslationEstimate(dy, dx, peak_ratio, True, "accepted")


def translate_image(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate without circular wrap-around artifacts."""

    return ndimage.shift(
        np.asarray(image, dtype=np.float32),
        shift=(float(dy), float(dx)),
        order=1,
        mode="nearest",
        prefilter=False,
    )


def estimate_cumulative_translations(
    images: dict[int, np.ndarray],
    masks: dict[int, list[Detection]] | None = None,
    config: StabilizationConfig = StabilizationConfig(),
) -> tuple[dict[int, tuple[float, float]], dict]:
    """Estimate cumulative shifts that align every frame with the first frame."""

    if not images:
        return {}, {"accepted": 0, "rejected": 0, "frames": {}}
    masks = masks or {}
    ordered = sorted(images)
    cumulative: dict[int, tuple[float, float]] = {ordered[0]: (0.0, 0.0)}
    diagnostics: dict[str, object] = {"accepted": 0, "rejected": 0, "frames": {}}
    anchor_frame = ordered[0]
    anchor = mask_detections(
        images[anchor_frame],
        masks.get(anchor_frame, ()),
        scale=config.mask_scale,
    )
    anchor_dy = 0.0
    anchor_dx = 0.0
    for frame in ordered[1:]:
        current = mask_detections(
            images[frame],
            masks.get(frame, ()),
            scale=config.mask_scale,
        )
        estimate = phase_translation(anchor, current, config)
        if estimate.accepted:
            total_dy = anchor_dy + estimate.dy
            total_dx = anchor_dx + estimate.dx
            diagnostics["accepted"] = int(diagnostics["accepted"]) + 1
        else:
            # Do not promote a rejected frame to the registration anchor.  Doing
            # so would permanently lose its unobserved displacement from every
            # later cumulative transform.
            total_dy = anchor_dy
            total_dx = anchor_dx
            diagnostics["rejected"] = int(diagnostics["rejected"]) + 1
        cumulative[frame] = (total_dy, total_dx)
        frame_diagnostics = diagnostics["frames"]
        assert isinstance(frame_diagnostics, dict)
        frame_diagnostics[str(frame)] = {
            "from_frame": anchor_frame,
            "dy": estimate.dy,
            "dx": estimate.dx,
            "peak_ratio": estimate.peak_ratio,
            "accepted": estimate.accepted,
            "reason": estimate.reason,
            "cumulative_dy": total_dy,
            "cumulative_dx": total_dx,
        }
        if estimate.accepted:
            anchor = current
            anchor_frame = frame
            anchor_dy = total_dy
            anchor_dx = total_dx
    return cumulative, diagnostics


def estimate_sequence_translations(
    image_paths: dict[int, Path],
    track_rows: Iterable[Detection],
    config: StabilizationConfig = StabilizationConfig(),
) -> tuple[dict[int, tuple[float, float]], dict]:
    """Stream a sequence from disk and estimate first-frame coordinates."""

    if not image_paths:
        return {}, {"accepted": 0, "rejected": 0, "frames": {}}
    frame_rows = group_frame(track_rows)
    ordered = sorted(image_paths)
    cumulative: dict[int, tuple[float, float]] = {ordered[0]: (0.0, 0.0)}
    diagnostics: dict[str, object] = {"accepted": 0, "rejected": 0, "frames": {}}
    anchor_frame = ordered[0]
    anchor = mask_detections(
        load_gray(image_paths[anchor_frame]),
        frame_rows.get(anchor_frame, ()),
        scale=config.mask_scale,
    )
    anchor_dy = 0.0
    anchor_dx = 0.0
    for frame in ordered[1:]:
        current = mask_detections(
            load_gray(image_paths[frame]),
            frame_rows.get(frame, ()),
            scale=config.mask_scale,
        )
        estimate = phase_translation(anchor, current, config)
        if estimate.accepted:
            total_dy = anchor_dy + estimate.dy
            total_dx = anchor_dx + estimate.dx
            diagnostics["accepted"] = int(diagnostics["accepted"]) + 1
        else:
            total_dy = anchor_dy
            total_dx = anchor_dx
            diagnostics["rejected"] = int(diagnostics["rejected"]) + 1
        cumulative[frame] = (total_dy, total_dx)
        frame_diagnostics = diagnostics["frames"]
        assert isinstance(frame_diagnostics, dict)
        frame_diagnostics[str(frame)] = {
            "from_frame": anchor_frame,
            "dy": estimate.dy,
            "dx": estimate.dx,
            "peak_ratio": estimate.peak_ratio,
            "accepted": estimate.accepted,
            "reason": estimate.reason,
            "cumulative_dy": total_dy,
            "cumulative_dx": total_dx,
        }
        if estimate.accepted:
            anchor = current
            anchor_frame = frame
            anchor_dy = total_dy
            anchor_dx = total_dx
    return cumulative, diagnostics


def make_stabilized_geometry(
    cumulative: dict[int, tuple[float, float]],
) -> Callable[[Detection], Detection]:
    """Create a geometry map for association while preserving source boxes."""

    def geometry(row: Detection) -> Detection:
        dy, dx = cumulative.get(row.frame_id, (0.0, 0.0))
        return replace(row, x1=row.x1 + dx, y1=row.y1 + dy)

    return geometry


def stabilize_directory(
    image_root: Path,
    tracks_dir: Path,
    output: Path,
    sequence_names: list[str] | None = None,
    config: StabilizationConfig = StabilizationConfig(),
) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    payload: dict[str, object] = {"config": config.__dict__, "sequences": {}}
    sequences = payload["sequences"]
    assert isinstance(sequences, dict)
    for track_file in prediction_files(tracks_dir):
        if allowed is not None and track_file.stem not in allowed:
            continue
        paths = image_index(image_root / track_file.stem)
        cumulative, diagnostics = estimate_sequence_translations(paths, read_rows(track_file), config)
        sequences[track_file.stem] = {
            "translations": {str(frame): [dy, dx] for frame, (dy, dx) in cumulative.items()},
            "diagnostics": diagnostics,
        }
    write_json(output, payload)
    return payload


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--max-shift", type=float, default=48.0)
    parser.add_argument("--min-peak-ratio", type=float, default=1.18)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = stabilize_directory(
        args.image_root,
        args.tracks_dir,
        args.output,
        _names(args.sequence_list),
        StabilizationConfig(max_shift=args.max_shift, min_peak_ratio=args.min_peak_ratio),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
