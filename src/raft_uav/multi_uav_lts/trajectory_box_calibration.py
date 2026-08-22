"""RTS trajectory smoothing and uncertainty-aware box calibration for LTS outputs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ._records import (
    Detection,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
)

_POLICY_SCHEMA = "raft-uav-multi-uav-lts-box-calibration-policy-v1"
_SUMMARY_SCHEMA = "raft-uav-multi-uav-lts-box-calibration-v1"
_EPS = 1e-9


@dataclass(frozen=True)
class BoxCalibrationParameters:
    """Parameters of the linear-Gaussian trajectory/box smoother."""

    process_accel_sigma: float = 0.50
    process_log_size_sigma: float = 0.035
    center_measurement_sigma: float = 0.30
    log_size_measurement_sigma: float = 0.12
    confidence_floor: float = 0.05
    recenter_weight: float = 1.0
    size_smoothing_weight: float = 0.75
    uncertainty_scale_x: float = 0.0
    uncertainty_scale_y: float = 0.0
    velocity_margin_scale: float = 0.0
    max_area_ratio: float = 4.0
    image_width: float | None = None
    image_height: float | None = None

    def validate(self) -> None:
        for name in (
            "process_accel_sigma",
            "process_log_size_sigma",
            "center_measurement_sigma",
            "log_size_measurement_sigma",
            "uncertainty_scale_x",
            "uncertainty_scale_y",
            "velocity_margin_scale",
        ):
            _nonnegative_finite(getattr(self, name), name=name)
        _unit(self.confidence_floor, name="confidence_floor", positive=True)
        _unit(self.recenter_weight, name="recenter_weight")
        _unit(self.size_smoothing_weight, name="size_smoothing_weight")
        _positive_finite(self.max_area_ratio, name="max_area_ratio")
        width = _optional_positive(self.image_width, name="image_width")
        height = _optional_positive(self.image_height, name="image_height")
        if (width is None) != (height is None):
            raise ValueError("image_width and image_height must be supplied together")


@dataclass(frozen=True)
class SequenceCalibrationSummary:
    sequence: str
    row_count: int
    track_count: int
    smoothed_track_count: int
    mean_center_shift_px: float
    max_center_shift_px: float
    mean_area_ratio: float
    max_area_ratio: float


@dataclass(frozen=True)
class BoxCalibrationSummary:
    schema: str
    prediction_path: str
    output_dir: str
    sequence_count: int
    row_count: int
    policy_path: str | None
    default_parameters: BoxCalibrationParameters
    sequences: tuple[SequenceCalibrationSummary, ...]


@dataclass(frozen=True)
class _SmoothedObservation:
    row: Detection
    state: np.ndarray
    covariance: np.ndarray


def calibrate_prediction_set(
    prediction_path: Path,
    output_dir: Path,
    *,
    parameters: BoxCalibrationParameters | None = None,
    policy_path: Path | None = None,
    sequences: Sequence[str] | None = None,
) -> BoxCalibrationSummary:
    """Smooth trajectories and calibrate boxes without using ground truth."""

    default = parameters or BoxCalibrationParameters()
    default.validate()
    policy = _load_policy(policy_path) if policy_path is not None else None
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

    summaries: list[SequenceCalibrationSummary] = []
    for name in sorted(texts):
        sequence = Path(name).stem
        if selected and sequence not in selected:
            continue
        rows = parse_detection_text(texts[name], source=f"{prediction_path}:{name}")
        reject_duplicate_keys(rows, label="prediction")
        sequence_parameters = _parameters_for_sequence(default, sequence, policy)
        calibrated, summary = calibrate_sequence(
            sequence,
            rows,
            parameters=sequence_parameters,
        )
        (output_dir / name).write_text(
            "".join(format_detection(row) + "\n" for row in calibrated),
            encoding="utf-8",
        )
        summaries.append(summary)

    return BoxCalibrationSummary(
        schema=_SUMMARY_SCHEMA,
        prediction_path=str(prediction_path),
        output_dir=str(output_dir),
        sequence_count=len(summaries),
        row_count=sum(summary.row_count for summary in summaries),
        policy_path=None if policy_path is None else str(policy_path),
        default_parameters=default,
        sequences=tuple(summaries),
    )


def calibrate_sequence(
    sequence: str,
    rows: Sequence[Detection],
    *,
    parameters: BoxCalibrationParameters,
) -> tuple[tuple[Detection, ...], SequenceCalibrationSummary]:
    """Calibrate one sequence, preserving frame/object identities exactly."""

    parameters.validate()
    grouped: dict[int, list[Detection]] = {}
    for row in rows:
        grouped.setdefault(row.object_id, []).append(row)

    calibrated: list[Detection] = []
    shifts: list[float] = []
    area_ratios: list[float] = []
    smoothed_tracks = 0
    for object_id in sorted(grouped):
        track = tuple(sorted(grouped[object_id], key=lambda row: row.frame_id))
        if len({row.frame_id for row in track}) != len(track):
            raise ValueError(
                f"{sequence}: duplicate frames for object identity {object_id}"
            )
        smoothed = _smooth_track(track, parameters)
        if len(track) >= 2:
            smoothed_tracks += 1
        for observation in smoothed:
            output = _calibrated_box(observation, parameters)
            calibrated.append(output)
            shifts.append(
                math.hypot(
                    output.center_x - observation.row.center_x,
                    output.center_y - observation.row.center_y,
                )
            )
            original_area = observation.row.width * observation.row.height
            area_ratios.append(output.width * output.height / original_area)

    calibrated.sort(key=lambda row: (row.frame_id, row.object_id))
    return tuple(calibrated), SequenceCalibrationSummary(
        sequence=sequence,
        row_count=len(calibrated),
        track_count=len(grouped),
        smoothed_track_count=smoothed_tracks,
        mean_center_shift_px=_mean(shifts),
        max_center_shift_px=max(shifts, default=0.0),
        mean_area_ratio=_mean(area_ratios, default=1.0),
        max_area_ratio=max(area_ratios, default=1.0),
    )


def _smooth_track(
    rows: Sequence[Detection],
    parameters: BoxCalibrationParameters,
) -> tuple[_SmoothedObservation, ...]:
    if not rows:
        return ()
    scale = float(np.median([math.sqrt(row.width * row.height) for row in rows]))
    scale = max(scale, 1.0)
    states: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    transitions: list[np.ndarray] = []

    first = rows[0]
    state = np.asarray(
        [first.center_x, first.center_y, 0.0, 0.0, math.log(first.width), math.log(first.height)],
        dtype=float,
    )
    if len(rows) >= 2:
        dt = max(1, rows[1].frame_id - rows[0].frame_id)
        state[2] = (rows[1].center_x - first.center_x) / dt
        state[3] = (rows[1].center_y - first.center_y) / dt
    first_r = _measurement_covariance(first, parameters)
    covariance = np.diag(
        [
            first_r[0, 0],
            first_r[1, 1],
            scale * scale,
            scale * scale,
            first_r[2, 2],
            first_r[3, 3],
        ]
    )

    for index, row in enumerate(rows):
        if index == 0:
            predicted_state = state.copy()
            predicted_covariance = covariance.copy()
            transition = np.eye(6, dtype=float)
        else:
            dt = row.frame_id - rows[index - 1].frame_id
            if dt <= 0:
                raise ValueError("track rows must have strictly increasing frame IDs")
            transition, process = _transition_and_process(dt, scale, parameters)
            predicted_state = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process

        measurement = np.asarray(
            [row.center_x, row.center_y, math.log(row.width), math.log(row.height)],
            dtype=float,
        )
        observation = _observation_matrix()
        noise = _measurement_covariance(row, parameters)
        innovation = measurement - observation @ predicted_state
        innovation_covariance = observation @ predicted_covariance @ observation.T + noise
        gain = predicted_covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
        state = predicted_state + gain @ innovation
        identity = np.eye(6, dtype=float)
        correction = identity - gain @ observation
        covariance = (
            correction @ predicted_covariance @ correction.T + gain @ noise @ gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)

        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        transitions.append(transition)
        states.append(state.copy())
        covariances.append(covariance.copy())

    smooth_states = [value.copy() for value in states]
    smooth_covariances = [value.copy() for value in covariances]
    for index in range(len(rows) - 2, -1, -1):
        transition = transitions[index + 1]
        predicted_covariance = predicted_covariances[index + 1]
        smoother_gain = (
            covariances[index]
            @ transition.T
            @ np.linalg.pinv(predicted_covariance)
        )
        smooth_states[index] = states[index] + smoother_gain @ (
            smooth_states[index + 1] - predicted_states[index + 1]
        )
        smooth_covariances[index] = covariances[index] + smoother_gain @ (
            smooth_covariances[index + 1] - predicted_covariance
        ) @ smoother_gain.T
        smooth_covariances[index] = 0.5 * (
            smooth_covariances[index] + smooth_covariances[index].T
        )

    return tuple(
        _SmoothedObservation(row, state, covariance)
        for row, state, covariance in zip(rows, smooth_states, smooth_covariances)
    )


def _transition_and_process(
    dt: int,
    scale: float,
    parameters: BoxCalibrationParameters,
) -> tuple[np.ndarray, np.ndarray]:
    delta = float(dt)
    transition = np.eye(6, dtype=float)
    transition[0, 2] = delta
    transition[1, 3] = delta
    process = np.zeros((6, 6), dtype=float)
    acceleration = parameters.process_accel_sigma * scale
    block = acceleration * acceleration * np.asarray(
        [
            [delta**4 / 4.0, delta**3 / 2.0],
            [delta**3 / 2.0, delta**2],
        ],
        dtype=float,
    )
    process[np.ix_((0, 2), (0, 2))] = block
    process[np.ix_((1, 3), (1, 3))] = block
    size_variance = parameters.process_log_size_sigma**2 * delta
    process[4, 4] = size_variance
    process[5, 5] = size_variance
    return transition, process


def _observation_matrix() -> np.ndarray:
    matrix = np.zeros((4, 6), dtype=float)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    matrix[2, 4] = 1.0
    matrix[3, 5] = 1.0
    return matrix


def _measurement_covariance(
    row: Detection,
    parameters: BoxCalibrationParameters,
) -> np.ndarray:
    confidence = max(parameters.confidence_floor, min(1.0, row.confidence))
    denominator = math.sqrt(confidence)
    sigma_x = parameters.center_measurement_sigma * max(row.width, 1.0) / denominator
    sigma_y = parameters.center_measurement_sigma * max(row.height, 1.0) / denominator
    sigma_size = parameters.log_size_measurement_sigma / denominator
    return np.diag(
        [sigma_x * sigma_x, sigma_y * sigma_y, sigma_size * sigma_size, sigma_size * sigma_size]
    )


def _calibrated_box(
    observation: _SmoothedObservation,
    parameters: BoxCalibrationParameters,
) -> Detection:
    row = observation.row
    state = observation.state
    covariance = observation.covariance
    center_x = _blend(row.center_x, float(state[0]), parameters.recenter_weight)
    center_y = _blend(row.center_y, float(state[1]), parameters.recenter_weight)
    smooth_width = math.exp(float(state[4]))
    smooth_height = math.exp(float(state[5]))
    base_width = _blend(row.width, smooth_width, parameters.size_smoothing_weight)
    base_height = _blend(row.height, smooth_height, parameters.size_smoothing_weight)
    sigma_x = math.sqrt(max(0.0, float(covariance[0, 0])))
    sigma_y = math.sqrt(max(0.0, float(covariance[1, 1])))
    margin_x = (
        parameters.uncertainty_scale_x * sigma_x
        + parameters.velocity_margin_scale * abs(float(state[2]))
    )
    margin_y = (
        parameters.uncertainty_scale_y * sigma_y
        + parameters.velocity_margin_scale * abs(float(state[3]))
    )
    width = max(_EPS, base_width + 2.0 * margin_x)
    height = max(_EPS, base_height + 2.0 * margin_y)
    width, height = _limit_area(width, height, row, parameters.max_area_ratio)
    x1 = center_x - 0.5 * width
    y1 = center_y - 0.5 * height
    x1, y1, width, height = _clip_box(x1, y1, width, height, parameters)
    return Detection(
        row.frame_id,
        row.object_id,
        x1,
        y1,
        width,
        height,
        row.confidence,
        row.class_id,
        row.visibility,
    )


def _limit_area(
    width: float,
    height: float,
    row: Detection,
    max_area_ratio: float,
) -> tuple[float, float]:
    original_area = row.width * row.height
    ratio = width * height / original_area
    if ratio <= max_area_ratio:
        return width, height
    shrink = math.sqrt(max_area_ratio / ratio)
    return width * shrink, height * shrink


def _clip_box(
    x1: float,
    y1: float,
    width: float,
    height: float,
    parameters: BoxCalibrationParameters,
) -> tuple[float, float, float, float]:
    if parameters.image_width is None or parameters.image_height is None:
        return x1, y1, width, height
    image_width = float(parameters.image_width)
    image_height = float(parameters.image_height)
    x2 = min(image_width, max(0.0, x1 + width))
    y2 = min(image_height, max(0.0, y1 + height))
    x1 = min(image_width, max(0.0, x1))
    y1 = min(image_height, max(0.0, y1))
    if x2 <= x1:
        x2 = min(image_width, x1 + _EPS)
        x1 = max(0.0, x2 - _EPS)
    if y2 <= y1:
        y2 = min(image_height, y1 + _EPS)
        y1 = max(0.0, y2 - _EPS)
    return x1, y1, x2 - x1, y2 - y1


def _load_policy(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read box-calibration policy: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != _POLICY_SCHEMA:
        raise ValueError("unsupported box-calibration policy schema")
    prefixes = payload.get("prefixes", {})
    default = payload.get("default", {})
    if not isinstance(prefixes, Mapping) or not isinstance(default, Mapping):
        raise ValueError("box-calibration policy default/prefixes must be objects")
    return payload


def _parameters_for_sequence(
    default: BoxCalibrationParameters,
    sequence: str,
    policy: Mapping[str, object] | None,
) -> BoxCalibrationParameters:
    if policy is None:
        return default
    overrides: dict[str, object] = {}
    raw_default = policy.get("default", {})
    if isinstance(raw_default, Mapping):
        overrides.update(raw_default)
    prefixes = policy.get("prefixes", {})
    matches: list[tuple[int, Mapping[str, object]]] = []
    if isinstance(prefixes, Mapping):
        for prefix, values in prefixes.items():
            key = str(prefix)
            if not isinstance(values, Mapping):
                raise ValueError(f"policy prefix {key!r} must map to an object")
            if sequence == key or sequence.startswith(key + "_"):
                matches.append((len(key), values))
    if matches:
        overrides.update(max(matches, key=lambda value: value[0])[1])
    allowed = set(BoxCalibrationParameters.__dataclass_fields__)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown box-calibration policy keys: {', '.join(unknown)}")
    parameters = replace(default, **overrides)
    parameters.validate()
    return parameters


def _validate_output_path(prediction_path: Path, output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    if prediction_path.is_dir():
        source = prediction_path.expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("output directory must not alias or be nested in predictions")


def write_summary(summary: BoxCalibrationSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    return default if not values else float(sum(values) / len(values))


def _blend(original: float, smoothed: float, weight: float) -> float:
    return (1.0 - weight) * original + weight * smoothed


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


def _optional_positive(value: object, *, name: str) -> float | None:
    return None if value is None else _positive_finite(value, name=name)


def _unit(value: object, *, name: str, positive: bool = False) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed > 1.0 or (positive and parsed <= 0.0):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--process-accel-sigma", type=float, default=0.50)
    parser.add_argument("--process-log-size-sigma", type=float, default=0.035)
    parser.add_argument("--center-measurement-sigma", type=float, default=0.30)
    parser.add_argument("--log-size-measurement-sigma", type=float, default=0.12)
    parser.add_argument("--confidence-floor", type=float, default=0.05)
    parser.add_argument("--recenter-weight", type=float, default=1.0)
    parser.add_argument("--size-smoothing-weight", type=float, default=0.75)
    parser.add_argument("--uncertainty-scale-x", type=float, default=0.0)
    parser.add_argument("--uncertainty-scale-y", type=float, default=0.0)
    parser.add_argument("--velocity-margin-scale", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=4.0)
    parser.add_argument("--image-width", type=float)
    parser.add_argument("--image-height", type=float)
    args = parser.parse_args(argv)
    parameters = BoxCalibrationParameters(
        process_accel_sigma=args.process_accel_sigma,
        process_log_size_sigma=args.process_log_size_sigma,
        center_measurement_sigma=args.center_measurement_sigma,
        log_size_measurement_sigma=args.log_size_measurement_sigma,
        confidence_floor=args.confidence_floor,
        recenter_weight=args.recenter_weight,
        size_smoothing_weight=args.size_smoothing_weight,
        uncertainty_scale_x=args.uncertainty_scale_x,
        uncertainty_scale_y=args.uncertainty_scale_y,
        velocity_margin_scale=args.velocity_margin_scale,
        max_area_ratio=args.max_area_ratio,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    summary = calibrate_prediction_set(
        args.prediction_path,
        args.output_dir,
        parameters=parameters,
        policy_path=args.policy_json,
        sequences=args.sequences,
    )
    if args.output_json is not None:
        write_summary(summary, args.output_json)
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
