"""Generate tiny-target proposals from motion-compensated temporal residuals."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage

from ._records import Detection, format_detection

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_SUMMARY_SCHEMA = "raft-uav-multi-uav-lts-temporal-residual-proposals-v1"


@dataclass(frozen=True)
class TemporalResidualParameters:
    """Controls for the detector-independent temporal proposal source."""

    registration_stride: int = 8
    max_registration_shift_px: float = 160.0
    residual_sigma_floor: float = 0.01
    smooth_sigma_px: float = 0.8
    z_threshold: float = 5.0
    min_component_area_px: int = 1
    max_component_area_px: int = 900
    box_padding_px: float = 3.0
    max_proposals_per_frame: int = 96
    bidirectional: bool = True

    def validate(self) -> None:
        _positive_int(self.registration_stride, name="registration_stride")
        _positive_finite(
            self.max_registration_shift_px,
            name="max_registration_shift_px",
        )
        _positive_finite(self.residual_sigma_floor, name="residual_sigma_floor")
        _nonnegative_finite(self.smooth_sigma_px, name="smooth_sigma_px")
        _positive_finite(self.z_threshold, name="z_threshold")
        minimum = _positive_int(
            self.min_component_area_px,
            name="min_component_area_px",
        )
        maximum = _positive_int(
            self.max_component_area_px,
            name="max_component_area_px",
        )
        if maximum < minimum:
            raise ValueError("max_component_area_px must be >= min_component_area_px")
        _nonnegative_finite(self.box_padding_px, name="box_padding_px")
        _positive_int(self.max_proposals_per_frame, name="max_proposals_per_frame")
        if not isinstance(self.bidirectional, bool):
            raise ValueError("bidirectional must be a Boolean")


@dataclass(frozen=True)
class TemporalResidualSequenceSummary:
    sequence: str
    frame_count: int
    proposal_count: int
    frames_with_proposals: int
    mean_proposals_per_nonempty_frame: float
    max_proposals_in_frame: int
    mean_registration_shift_px: float
    max_registration_shift_px: float


@dataclass(frozen=True)
class TemporalResidualSummary:
    schema: str
    sequence_root: str
    output_dir: str
    parameters: TemporalResidualParameters
    sequence_count: int
    frame_count: int
    proposal_count: int
    sequences: tuple[TemporalResidualSequenceSummary, ...]


def generate_temporal_residual_proposals(
    sequence_root: Path,
    output_dir: Path,
    *,
    parameters: TemporalResidualParameters | None = None,
    sequences: Sequence[str] | None = None,
) -> TemporalResidualSummary:
    """Generate one LTS proposal file per image sequence."""

    config = parameters or TemporalResidualParameters()
    config.validate()
    if not sequence_root.is_dir():
        raise NotADirectoryError(sequence_root)
    _validate_output_path(sequence_root, output_dir)
    available = {
        path.name: path
        for path in sorted(sequence_root.iterdir())
        if path.is_dir()
    }
    requested = tuple(dict.fromkeys(str(value) for value in (sequences or ())))
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"unknown image sequences: {', '.join(missing)}")
        selected = requested
    else:
        selected = tuple(available)
    if not selected:
        raise ValueError(f"sequence root contains no sequence directories: {sequence_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()

    summaries: list[TemporalResidualSequenceSummary] = []
    for sequence in selected:
        frame_paths = _frame_paths(available[sequence])
        rows, summary = _process_sequence(sequence, frame_paths, config)
        (output_dir / f"{sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        summaries.append(summary)

    return TemporalResidualSummary(
        schema=_SUMMARY_SCHEMA,
        sequence_root=str(sequence_root),
        output_dir=str(output_dir),
        parameters=config,
        sequence_count=len(summaries),
        frame_count=sum(row.frame_count for row in summaries),
        proposal_count=sum(row.proposal_count for row in summaries),
        sequences=tuple(summaries),
    )


def _process_sequence(
    sequence: str,
    frame_paths: Sequence[Path],
    parameters: TemporalResidualParameters,
) -> tuple[tuple[Detection, ...], TemporalResidualSequenceSummary]:
    if not frame_paths:
        raise ValueError(f"{sequence}: no image frames")
    rows: list[Detection] = []
    proposal_counts: list[int] = []
    registration_shifts: list[float] = []

    previous: np.ndarray | None = None
    current = _read_gray(frame_paths[0])
    following = _read_gray(frame_paths[1]) if len(frame_paths) > 1 else None
    shape = current.shape
    for frame_index in range(len(frame_paths)):
        if current.shape != shape:
            raise ValueError(f"{sequence}: image dimensions change within the sequence")
        if following is not None and following.shape != shape:
            raise ValueError(f"{sequence}: image dimensions change within the sequence")
        proposals, shifts = temporal_residual_frame(
            previous,
            current,
            following,
            parameters=parameters,
        )
        frame_id = frame_index + 1
        proposal_counts.append(len(proposals))
        registration_shifts.extend(math.hypot(dy, dx) for dy, dx in shifts)
        for proposal_id, (x1, y1, width, height, confidence) in enumerate(
            proposals,
            start=1,
        ):
            rows.append(
                Detection(
                    frame_id,
                    proposal_id,
                    x1,
                    y1,
                    width,
                    height,
                    confidence,
                    1,
                    1.0,
                )
            )
        previous = current
        current = following if following is not None else current
        next_index = frame_index + 2
        following = (
            _read_gray(frame_paths[next_index])
            if next_index < len(frame_paths)
            else None
        )

    nonempty = [value for value in proposal_counts if value > 0]
    return tuple(rows), TemporalResidualSequenceSummary(
        sequence=sequence,
        frame_count=len(frame_paths),
        proposal_count=len(rows),
        frames_with_proposals=len(nonempty),
        mean_proposals_per_nonempty_frame=(
            0.0 if not nonempty else float(sum(nonempty) / len(nonempty))
        ),
        max_proposals_in_frame=max(proposal_counts, default=0),
        mean_registration_shift_px=(
            0.0
            if not registration_shifts
            else float(sum(registration_shifts) / len(registration_shifts))
        ),
        max_registration_shift_px=max(registration_shifts, default=0.0),
    )


def temporal_residual_frame(
    previous: np.ndarray | None,
    current: np.ndarray,
    following: np.ndarray | None,
    *,
    parameters: TemporalResidualParameters,
) -> tuple[tuple[tuple[float, float, float, float, float], ...], tuple[tuple[float, float], ...]]:
    """Return proposal boxes and applied neighbour-to-current registration shifts."""

    parameters.validate()
    current_gray = _validate_gray(current, name="current")
    neighbors: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    for name, neighbor in (("previous", previous), ("following", following)):
        if neighbor is None:
            continue
        if name == "following" and not parameters.bidirectional and previous is not None:
            continue
        neighbor_gray = _validate_gray(neighbor, name=name)
        if neighbor_gray.shape != current_gray.shape:
            raise ValueError(f"{name} frame shape differs from current frame")
        shift = estimate_translation(
            current_gray,
            neighbor_gray,
            stride=parameters.registration_stride,
        )
        if math.hypot(*shift) > parameters.max_registration_shift_px:
            continue
        aligned = _translate_with_nan(neighbor_gray, shift[0], shift[1])
        neighbors.append(aligned)
        shifts.append(shift)
    if not neighbors:
        return (), tuple(shifts)

    stack = np.stack(neighbors, axis=0)
    with np.errstate(invalid="ignore"):
        reference = np.nanmedian(stack, axis=0)
    valid = np.isfinite(reference)
    residual = np.zeros_like(current_gray, dtype=float)
    residual[valid] = np.abs(current_gray[valid] - reference[valid])
    valid_values = residual[valid]
    if valid_values.size == 0:
        return (), tuple(shifts)
    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    robust_sigma = max(parameters.residual_sigma_floor, 1.4826 * mad)
    z = np.zeros_like(residual)
    z[valid] = np.maximum(0.0, residual[valid] - median) / robust_sigma
    if parameters.smooth_sigma_px > 0.0:
        z = ndimage.gaussian_filter(z, parameters.smooth_sigma_px, mode="nearest")
    active = valid & (z >= parameters.z_threshold)
    labels, component_count = ndimage.label(active)
    components: list[tuple[float, tuple[float, float, float, float, float]]] = []
    height_px, width_px = current_gray.shape
    for label_index in range(1, component_count + 1):
        yy, xx = np.nonzero(labels == label_index)
        area = int(xx.size)
        if area < parameters.min_component_area_px:
            continue
        if area > parameters.max_component_area_px:
            continue
        peak = float(np.max(z[yy, xx]))
        padding = parameters.box_padding_px
        x1 = max(0.0, float(xx.min()) - padding)
        y1 = max(0.0, float(yy.min()) - padding)
        x2 = min(float(width_px), float(xx.max() + 1) + padding)
        y2 = min(float(height_px), float(yy.max() + 1) + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        excess = max(0.0, peak - parameters.z_threshold)
        confidence = max(0.001, min(1.0, 1.0 - math.exp(-excess / 4.0)))
        components.append((peak, (x1, y1, x2 - x1, y2 - y1, confidence)))
    components.sort(
        key=lambda value: (
            -value[0],
            value[1][1],
            value[1][0],
            value[1][2],
            value[1][3],
        )
    )
    return (
        tuple(value[1] for value in components[: parameters.max_proposals_per_frame]),
        tuple(shifts),
    )


def estimate_translation(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    stride: int = 8,
) -> tuple[float, float]:
    """Estimate the integer translation to apply to ``moving`` to align it to reference."""

    step = _positive_int(stride, name="stride")
    reference_gray = _validate_gray(reference, name="reference")[::step, ::step]
    moving_gray = _validate_gray(moving, name="moving")[::step, ::step]
    if reference_gray.shape != moving_gray.shape:
        raise ValueError("reference and moving frames must have the same shape")
    if min(reference_gray.shape) < 2:
        return 0.0, 0.0
    window_y = np.hanning(reference_gray.shape[0])
    window_x = np.hanning(reference_gray.shape[1])
    window = window_y[:, None] * window_x[None, :]
    left = (reference_gray - float(reference_gray.mean())) * window
    right = (moving_gray - float(moving_gray.mean())) * window
    left_fft = np.fft.fft2(left)
    right_fft = np.fft.fft2(right)
    cross = left_fft * np.conj(right_fft)
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-12)
    correlation = np.fft.ifft2(cross).real
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift_y = int(peak_y)
    shift_x = int(peak_x)
    if shift_y > correlation.shape[0] // 2:
        shift_y -= correlation.shape[0]
    if shift_x > correlation.shape[1] // 2:
        shift_x -= correlation.shape[1]
    return float(shift_y * step), float(shift_x * step)


def _translate_with_nan(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    shift_y = int(round(dy))
    shift_x = int(round(dx))
    height, width = image.shape
    result = np.full((height, width), np.nan, dtype=float)
    source_y0 = max(0, -shift_y)
    source_y1 = min(height, height - shift_y)
    source_x0 = max(0, -shift_x)
    source_x1 = min(width, width - shift_x)
    target_y0 = source_y0 + shift_y
    target_y1 = source_y1 + shift_y
    target_x0 = source_x0 + shift_x
    target_x1 = source_x1 + shift_x
    if source_y1 > source_y0 and source_x1 > source_x0:
        result[target_y0:target_y1, target_x0:target_x1] = image[
            source_y0:source_y1,
            source_x0:source_x1,
        ]
    return result


def _frame_paths(sequence_dir: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            path
            for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"sequence contains no supported images: {sequence_dir}")
    return paths


def _read_gray(path: Path) -> np.ndarray:
    image = np.asarray(mpimg.imread(path))
    if image.ndim == 3:
        if image.shape[2] < 3:
            image = image[..., 0]
        else:
            image = (
                0.2126 * image[..., 0]
                + 0.7152 * image[..., 1]
                + 0.0722 * image[..., 2]
            )
    if image.ndim != 2:
        raise ValueError(f"{path}: unsupported image shape {image.shape}")
    gray = np.asarray(image, dtype=float)
    if gray.size == 0 or not np.isfinite(gray).all():
        raise ValueError(f"{path}: image must be non-empty and finite")
    minimum = float(gray.min())
    maximum = float(gray.max())
    if minimum < 0.0:
        raise ValueError(f"{path}: image intensities must be non-negative")
    if maximum > 1.0:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0)


def _validate_gray(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} frame must be a finite non-empty 2D array")
    return array


def _validate_output_path(sequence_root: Path, output_dir: Path) -> None:
    source = sequence_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("output directory must not alias or be nested in sequence_root")


def write_summary(summary: TemporalResidualSummary, path: Path) -> None:
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


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--registration-stride", type=int, default=8)
    parser.add_argument("--max-registration-shift-px", type=float, default=160.0)
    parser.add_argument("--residual-sigma-floor", type=float, default=0.01)
    parser.add_argument("--smooth-sigma-px", type=float, default=0.8)
    parser.add_argument("--z-threshold", type=float, default=5.0)
    parser.add_argument("--min-component-area-px", type=int, default=1)
    parser.add_argument("--max-component-area-px", type=int, default=900)
    parser.add_argument("--box-padding-px", type=float, default=3.0)
    parser.add_argument("--max-proposals-per-frame", type=int, default=96)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args(argv)
    parameters = TemporalResidualParameters(
        registration_stride=args.registration_stride,
        max_registration_shift_px=args.max_registration_shift_px,
        residual_sigma_floor=args.residual_sigma_floor,
        smooth_sigma_px=args.smooth_sigma_px,
        z_threshold=args.z_threshold,
        min_component_area_px=args.min_component_area_px,
        max_component_area_px=args.max_component_area_px,
        box_padding_px=args.box_padding_px,
        max_proposals_per_frame=args.max_proposals_per_frame,
        bidirectional=not args.forward_only,
    )
    summary = generate_temporal_residual_proposals(
        args.sequence_root,
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
