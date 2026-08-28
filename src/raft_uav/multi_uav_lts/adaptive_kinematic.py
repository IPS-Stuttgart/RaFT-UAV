"""Area-adaptive robust kinematic smoother used as a lightweight AKKF control."""

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
class AdaptiveKinematicConfig:
    process_noise: float = 1.4
    measurement_noise: float = 3.0
    size_noise: float = 0.09
    student_df: float = 5.0
    tiny_reference_side: float = 18.0
    min_velocity_damping: float = 0.68
    max_velocity_damping: float = 0.97
    confidence_noise_gain: float = 2.0
    perspective_brake: float = 0.12
    minimum_robust_weight: float = 0.08


def _observation(row: Detection) -> np.ndarray:
    return np.asarray(
        [
            row.center_x,
            row.center_y,
            math.log(max(row.width, 1e-6)),
            math.log(max(row.height, 1e-6)),
        ],
        dtype=float,
    )


def _damping(
    row: Detection,
    previous: Detection | None,
    image_height: int | None,
    config: AdaptiveKinematicConfig,
) -> float:
    side = math.sqrt(max(row.width * row.height, 1e-6))
    side_fraction = side / (side + config.tiny_reference_side)
    damping = config.min_velocity_damping + (
        config.max_velocity_damping - config.min_velocity_damping
    ) * side_fraction
    if previous is not None and image_height and image_height > 0:
        vertical_velocity = row.center_y - previous.center_y
        size_velocity = math.log(max(row.width * row.height, 1e-6)) - math.log(
            max(previous.width * previous.height, 1e-6)
        )
        toward_horizon = vertical_velocity < 0.0 and size_velocity < 0.0
        if toward_horizon:
            horizon_strength = 1.0 - float(np.clip(row.center_y / image_height, 0.0, 1.0))
            damping *= 1.0 - config.perspective_brake * horizon_strength
    return float(np.clip(damping, config.min_velocity_damping, config.max_velocity_damping))


def smooth_track(
    rows: list[Detection],
    config: AdaptiveKinematicConfig = AdaptiveKinematicConfig(),
    image_shape: tuple[int, int] | None = None,
) -> list[Detection]:
    track = sorted(rows, key=lambda row: row.frame_id)
    if len(track) < 2:
        return track
    observation_matrix = np.zeros((4, 6), dtype=float)
    observation_matrix[0, 0] = 1.0
    observation_matrix[1, 1] = 1.0
    observation_matrix[2, 4] = 1.0
    observation_matrix[3, 5] = 1.0
    initial = _observation(track[0])
    state = np.asarray([initial[0], initial[1], 0.0, 0.0, initial[2], initial[3]])
    covariance = np.diag([16.0, 16.0, 25.0, 25.0, 0.2, 0.2])
    filtered_states = []
    filtered_covariances = []
    predicted_states = []
    predicted_covariances = []
    transition_matrices = []
    previous_frame = track[0].frame_id
    previous_row: Detection | None = None

    for row in track:
        dt = max(1, row.frame_id - previous_frame)
        previous_frame = row.frame_id
        image_height = None if image_shape is None else image_shape[0]
        damping = _damping(row, previous_row, image_height, config)
        transition = np.eye(6)
        transition[0, 2] = dt
        transition[1, 3] = dt
        transition[2, 2] = damping**dt
        transition[3, 3] = damping**dt
        side = math.sqrt(max(row.width * row.height, 1e-6))
        tiny_factor = max(config.tiny_reference_side / max(side, 2.0), 0.5)
        process = np.diag(
            [
                dt**3,
                dt**3,
                dt**2,
                dt**2,
                0.02 * dt,
                0.02 * dt,
            ]
        ) * config.process_noise * tiny_factor
        predicted_state = transition @ state
        predicted_covariance = transition @ covariance @ transition.T + process

        confidence = float(np.clip(row.confidence, 0.0, 1.0))
        confidence_factor = 1.0 + config.confidence_noise_gain * (1.0 - confidence)
        measurement = np.diag(
            [
                (config.measurement_noise * tiny_factor * confidence_factor) ** 2,
                (config.measurement_noise * tiny_factor * confidence_factor) ** 2,
                (config.size_noise * confidence_factor) ** 2,
                (config.size_noise * confidence_factor) ** 2,
            ]
        )
        innovation = _observation(row) - observation_matrix @ predicted_state
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T + measurement
        )
        inverse = np.linalg.pinv(innovation_covariance)
        mahalanobis = float(innovation @ inverse @ innovation)
        robust_weight = (config.student_df + len(innovation)) / (
            config.student_df + mahalanobis
        )
        robust_weight = max(robust_weight, config.minimum_robust_weight)
        effective_measurement = measurement / robust_weight
        innovation_covariance = (
            observation_matrix @ predicted_covariance @ observation_matrix.T
            + effective_measurement
        )
        gain = (
            predicted_covariance
            @ observation_matrix.T
            @ np.linalg.pinv(innovation_covariance)
        )
        state = predicted_state + gain @ innovation
        covariance = (np.eye(6) - gain @ observation_matrix) @ predicted_covariance
        covariance = 0.5 * (covariance + covariance.T)

        predicted_states.append(predicted_state)
        predicted_covariances.append(predicted_covariance)
        transition_matrices.append(transition)
        filtered_states.append(state.copy())
        filtered_covariances.append(covariance.copy())
        previous_row = row

    smoothed_states = [value.copy() for value in filtered_states]
    smoothed_covariances = [value.copy() for value in filtered_covariances]
    for index in range(len(track) - 2, -1, -1):
        transition = transition_matrices[index + 1]
        gain = (
            filtered_covariances[index]
            @ transition.T
            @ np.linalg.pinv(predicted_covariances[index + 1])
        )
        smoothed_states[index] = filtered_states[index] + gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covariances[index] = filtered_covariances[index] + gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ gain.T

    output = []
    for row, smoothed in zip(track, smoothed_states, strict=True):
        width = float(np.exp(np.clip(smoothed[4], -5.0, 12.0)))
        height = float(np.exp(np.clip(smoothed[5], -5.0, 12.0)))
        candidate = replace(
            row,
            x1=float(smoothed[0] - 0.5 * width),
            y1=float(smoothed[1] - 0.5 * height),
            width=width,
            height=height,
        )
        if image_shape is not None:
            candidate = clip_box(candidate, image_shape[1], image_shape[0])
        else:
            candidate = replace(
                candidate,
                x1=max(0.0, candidate.x1),
                y1=max(0.0, candidate.y1),
            )
        output.append(candidate)
    return output


def smooth_directory(
    input_dir: Path,
    output_dir: Path,
    sequence_names: list[str] | None = None,
    config: AdaptiveKinematicConfig = AdaptiveKinematicConfig(),
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
        tracks = group_id(read_rows(file))
        rows = [
            row
            for identity in sorted(tracks)
            for row in smooth_track(tracks[identity], config, image_shape)
        ]
        write_rows(output_dir / file.name, rows)
        summary[file.stem] = len(rows)
    return {"sequences": summary, "config": config.__dict__}


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--process-noise", type=float, default=1.4)
    parser.add_argument("--measurement-noise", type=float, default=3.0)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = smooth_directory(
        args.input_dir,
        args.output_dir,
        _names(args.sequence_list),
        AdaptiveKinematicConfig(
            process_noise=args.process_noise,
            measurement_noise=args.measurement_noise,
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
