"""Seed-anchored swarm-geometry reassociation with bidirectional consensus."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._full_stack_io import group_frame, prediction_files, read_rows, write_json, write_rows
from ._records import Detection

Geometry = Callable[[Detection], Detection]


@dataclass(frozen=True)
class FormationConfig:
    max_gap: int = 18
    motion_scale: float = 3.0
    motion_gate: float = 12.0
    size_scale: float = 0.8
    missing_cost: float = 2.4
    label_inertia: float = 0.04
    confidence_reward: float = 0.12
    formation_weight: float = 0.45
    formation_ema: float = 0.04
    formation_clip: float = 0.35
    huber_delta: float = 0.30
    max_swap_passes: int = 4
    bidirectional_switch_penalty: float = 0.08
    include_births: bool = False


def _side(row: Detection) -> float:
    return max(math.sqrt(max(row.width * row.height, 1e-6)), 2.0)


def _predict(
    history: list[Detection],
    frame: int,
    geometry: Geometry | None = None,
) -> tuple[float, float, float, float, int]:
    latest_source = history[-1]
    latest = latest_source if geometry is None else geometry(latest_source)
    gap = abs(frame - latest.frame_id)
    center_x = latest.center_x
    center_y = latest.center_y
    if len(history) >= 2:
        previous_source = history[-2]
        previous = previous_source if geometry is None else geometry(previous_source)
        denominator = latest.frame_id - previous.frame_id
        if denominator != 0:
            ratio = (frame - latest.frame_id) / denominator
            center_x += ratio * (latest.center_x - previous.center_x)
            center_y += ratio * (latest.center_y - previous.center_y)
    recent = [row if geometry is None else geometry(row) for row in history[-3:]]
    widths = [row.width for row in recent]
    heights = [row.height for row in recent]
    return center_x, center_y, float(np.median(widths)), float(np.median(heights)), gap


def _huber(value: float, delta: float) -> float:
    absolute = abs(value)
    if absolute <= delta:
        return 0.5 * value * value
    return delta * (absolute - 0.5 * delta)


def _initial_pair_state(
    seeds: dict[int, Detection],
    geometry: Geometry | None = None,
) -> dict[tuple[int, int], float]:
    state = {}
    identities = sorted(seeds)
    for left_index, left_id in enumerate(identities):
        for right_id in identities[left_index + 1 :]:
            left_source = seeds[left_id]
            right_source = seeds[right_id]
            left = left_source if geometry is None else geometry(left_source)
            right = right_source if geometry is None else geometry(right_source)
            normalization = 0.5 * (_side(left) + _side(right))
            state[(left_id, right_id)] = math.hypot(
                left.center_x - right.center_x,
                left.center_y - right.center_y,
            ) / normalization
    return state


def _unary_cost(
    identity: int,
    detection: Detection,
    history: list[Detection],
    config: FormationConfig,
    geometry: Geometry | None = None,
) -> float:
    center_x, center_y, width, height, gap = _predict(
        history,
        detection.frame_id,
        geometry,
    )
    geometric_detection = detection if geometry is None else geometry(detection)
    scale = max(math.sqrt(width * height), _side(geometric_detection), 2.0)
    normalized_motion = math.hypot(
        geometric_detection.center_x - center_x,
        geometric_detection.center_y - center_y,
    ) / (scale * math.sqrt(max(gap, 1)))
    if gap > config.max_gap or normalized_motion > config.motion_gate:
        return config.missing_cost + 100.0 + normalized_motion
    size_delta = abs(
        math.log(max(geometric_detection.width, 1e-6) / max(width, 1e-6))
    ) + abs(
        math.log(max(geometric_detection.height, 1e-6) / max(height, 1e-6))
    )
    return (
        (normalized_motion / config.motion_scale) ** 2
        + (size_delta / config.size_scale) ** 2
        + config.label_inertia * float(detection.object_id != identity)
        - config.confidence_reward * float(np.clip(detection.confidence, 0.0, 1.0))
    )


def _formation_cost(
    assignment: dict[int, int],
    detections: list[Detection],
    pair_state: dict[tuple[int, int], float],
    config: FormationConfig,
    geometry: Geometry | None = None,
) -> float:
    identities = sorted(assignment)
    total = 0.0
    count = 0
    for left_index, left_id in enumerate(identities):
        for right_id in identities[left_index + 1 :]:
            key = (min(left_id, right_id), max(left_id, right_id))
            expected = pair_state.get(key)
            if expected is None or expected <= 1e-6:
                continue
            left_source = detections[assignment[left_id]]
            right_source = detections[assignment[right_id]]
            left = left_source if geometry is None else geometry(left_source)
            right = right_source if geometry is None else geometry(right_source)
            observed = math.hypot(
                left.center_x - right.center_x,
                left.center_y - right.center_y,
            ) / (0.5 * (_side(left) + _side(right)))
            relative_error = (observed - expected) / max(expected, 1.0)
            total += _huber(relative_error, config.huber_delta)
            count += 1
    return 0.0 if count == 0 else config.formation_weight * total / count


def _objective_with_indices(
    assignment: dict[int, int],
    identity_indices: dict[int, int],
    unary: np.ndarray,
    detections: list[Detection],
    pair_state: dict[tuple[int, int], float],
    config: FormationConfig,
    geometry: Geometry | None = None,
) -> float:
    value = sum(
        unary[identity_indices[identity], index]
        for identity, index in assignment.items()
    )
    return float(value) + _formation_cost(
        assignment,
        detections,
        pair_state,
        config,
        geometry,
    )


def _swap_refine(
    assignment: dict[int, int],
    identity_indices: dict[int, int],
    unary: np.ndarray,
    detections: list[Detection],
    pair_state: dict[tuple[int, int], float],
    config: FormationConfig,
    geometry: Geometry | None = None,
) -> tuple[dict[int, int], int]:
    current = dict(assignment)
    current_cost = _objective_with_indices(
        current,
        identity_indices,
        unary,
        detections,
        pair_state,
        config,
        geometry,
    )
    swaps = 0
    for _ in range(config.max_swap_passes):
        best_cost = current_cost
        best_pair: tuple[int, int] | None = None
        identities = sorted(current)
        for left_index, left_id in enumerate(identities):
            for right_id in identities[left_index + 1 :]:
                candidate = dict(current)
                candidate[left_id], candidate[right_id] = candidate[right_id], candidate[left_id]
                candidate_cost = _objective_with_indices(
                    candidate,
                    identity_indices,
                    unary,
                    detections,
                    pair_state,
                    config,
                    geometry,
                )
                if candidate_cost + 1e-12 < best_cost:
                    best_cost = candidate_cost
                    best_pair = (left_id, right_id)
        if best_pair is None:
            break
        left_id, right_id = best_pair
        current[left_id], current[right_id] = current[right_id], current[left_id]
        current_cost = best_cost
        swaps += 1
    return current, swaps


def _update_pair_state(
    assignment: dict[int, int],
    detections: list[Detection],
    pair_state: dict[tuple[int, int], float],
    config: FormationConfig,
    geometry: Geometry | None = None,
) -> None:
    identities = sorted(assignment)
    for left_index, left_id in enumerate(identities):
        for right_id in identities[left_index + 1 :]:
            key = (left_id, right_id)
            left_source = detections[assignment[left_id]]
            right_source = detections[assignment[right_id]]
            left = left_source if geometry is None else geometry(left_source)
            right = right_source if geometry is None else geometry(right_source)
            observed = math.hypot(
                left.center_x - right.center_x,
                left.center_y - right.center_y,
            ) / (0.5 * (_side(left) + _side(right)))
            previous = pair_state.get(key, observed)
            lower = previous * (1.0 - config.formation_clip)
            upper = previous * (1.0 + config.formation_clip)
            clipped = float(np.clip(observed, lower, upper))
            pair_state[key] = (1.0 - config.formation_ema) * previous + config.formation_ema * clipped


def refine_sequence(
    rows: list[Detection],
    seeds: list[Detection],
    config: FormationConfig = FormationConfig(),
    *,
    direction: int = 1,
    geometry: Geometry | None = None,
) -> tuple[list[Detection], dict]:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    seed_map = {row.object_id: row for row in seeds}
    if len(seed_map) != len(seeds):
        raise ValueError("duplicate seed identities")
    seed_ids = sorted(seed_map)
    by_frame = group_frame(rows)
    frames = sorted(by_frame)
    if not frames:
        return list(seeds), {"frames": 0, "relabels": 0, "swaps": 0, "direction": direction}

    pair_state = _initial_pair_state(seed_map, geometry)
    histories: dict[int, list[Detection]] = {identity: [seed_map[identity]] for identity in seed_ids}
    output_by_frame: dict[int, list[Detection]] = {1: list(seeds)}
    if direction < 0:
        last_frame = max(frames)
        terminal = {
            row.object_id: row
            for row in by_frame.get(last_frame, ())
            if row.object_id in seed_map
        }
        for identity in seed_ids:
            if identity in terminal:
                histories[identity] = [terminal[identity]]
        output_by_frame[last_frame] = [terminal[identity] for identity in sorted(terminal)]

    ordered_frames = [frame for frame in frames if frame != 1]
    if direction < 0:
        ordered_frames = list(reversed(ordered_frames))
        if ordered_frames and ordered_frames[0] == max(frames):
            ordered_frames = ordered_frames[1:]

    relabels = 0
    swaps = 0
    missing_assignments = 0
    for frame in ordered_frames:
        frame_rows = list(by_frame.get(frame, ()))
        seed_detections = [row for row in frame_rows if row.object_id in seed_map]
        if config.include_births:
            detections = frame_rows
        else:
            detections = seed_detections
        births = [row for row in frame_rows if row.object_id not in seed_map]
        if not detections:
            output_by_frame[frame] = births
            missing_assignments += len(seed_ids)
            continue

        identity_indices = {identity: index for index, identity in enumerate(seed_ids)}
        unary = np.empty((len(seed_ids), len(detections)), dtype=float)
        for identity, identity_index in identity_indices.items():
            for detection_index, detection in enumerate(detections):
                unary[identity_index, detection_index] = _unary_cost(
                    identity,
                    detection,
                    histories[identity],
                    config,
                    geometry,
                )
        augmented = np.full(
            (len(seed_ids), len(detections) + len(seed_ids)),
            config.missing_cost,
            dtype=float,
        )
        augmented[:, : len(detections)] = unary
        row_indices, column_indices = linear_sum_assignment(augmented)
        assignment = {
            seed_ids[row_index]: int(column_index)
            for row_index, column_index in zip(row_indices, column_indices, strict=True)
            if column_index < len(detections)
        }
        missing_assignments += len(seed_ids) - len(assignment)
        assignment, frame_swaps = _swap_refine(
            assignment,
            identity_indices,
            unary,
            detections,
            pair_state,
            config,
            geometry,
        )
        swaps += frame_swaps
        assigned_rows: list[Detection] = []
        for identity, detection_index in sorted(assignment.items()):
            source = detections[detection_index]
            if source.object_id != identity:
                relabels += 1
            assigned = replace(source, object_id=identity)
            assigned_rows.append(assigned)
            histories[identity].append(assigned)
            histories[identity].sort(key=lambda row: direction * row.frame_id)
        _update_pair_state(assignment, detections, pair_state, config, geometry)
        used = set(assignment.values())
        unassigned = [row for index, row in enumerate(detections) if index not in used]
        if not config.include_births:
            unassigned = []
        output_by_frame[frame] = [*assigned_rows, *births, *unassigned]

    output = [
        row
        for frame in sorted(output_by_frame)
        for row in sorted(output_by_frame[frame], key=lambda value: value.object_id)
    ]
    return output, {
        "frames": len(output_by_frame),
        "relabels": relabels,
        "swaps": swaps,
        "missing_assignments": missing_assignments,
        "direction": direction,
        "seed_count": len(seed_ids),
        "stabilized_geometry": geometry is not None,
    }


def _transition_cost(
    left: list[Detection],
    right: list[Detection],
    geometry: Geometry | None = None,
) -> float:
    left_map = {row.object_id: row for row in left}
    right_map = {row.object_id: row for row in right}
    identities = sorted(set(left_map) | set(right_map))
    total = 0.0
    count = 0
    for identity in identities:
        if identity not in left_map or identity not in right_map:
            total += 0.8
            count += 1
            continue
        first_source = left_map[identity]
        second_source = right_map[identity]
        first = first_source if geometry is None else geometry(first_source)
        second = second_source if geometry is None else geometry(second_source)
        gap = max(1, second.frame_id - first.frame_id)
        scale = max(_side(first), _side(second), 2.0)
        total += math.hypot(
            second.center_x - first.center_x,
            second.center_y - first.center_y,
        ) / (scale * math.sqrt(gap))
        total += 0.2 * abs(math.log(max(second.width, 1e-6) / max(first.width, 1e-6)))
        total += 0.2 * abs(math.log(max(second.height, 1e-6) / max(first.height, 1e-6)))
        count += 1
    return 0.0 if count == 0 else total / count


def bidirectional_refine_sequence(
    rows: list[Detection],
    seeds: list[Detection],
    config: FormationConfig = FormationConfig(),
    geometry: Geometry | None = None,
) -> tuple[list[Detection], dict]:
    forward, forward_diagnostics = refine_sequence(
        rows,
        seeds,
        config,
        direction=1,
        geometry=geometry,
    )
    backward, backward_diagnostics = refine_sequence(
        rows,
        seeds,
        config,
        direction=-1,
        geometry=geometry,
    )
    forward_frames = group_frame(forward)
    backward_frames = group_frame(backward)
    frames = sorted(set(forward_frames) | set(backward_frames))
    if not frames:
        return [], {
            "forward": forward_diagnostics,
            "backward": backward_diagnostics,
            "state_switches": 0,
        }

    costs = np.full((len(frames), 2), np.inf)
    parent = np.full((len(frames), 2), -1, dtype=int)
    costs[0] = 0.0
    states = (forward_frames, backward_frames)
    for frame_index in range(1, len(frames)):
        previous_frame = frames[frame_index - 1]
        frame = frames[frame_index]
        for current_state in range(2):
            current_rows = states[current_state].get(frame, [])
            for previous_state in range(2):
                previous_rows = states[previous_state].get(previous_frame, [])
                transition = _transition_cost(
                    previous_rows,
                    current_rows,
                    geometry,
                )
                if current_state != previous_state:
                    transition += config.bidirectional_switch_penalty
                candidate = costs[frame_index - 1, previous_state] + transition
                if candidate < costs[frame_index, current_state]:
                    costs[frame_index, current_state] = candidate
                    parent[frame_index, current_state] = previous_state
    state = int(np.argmin(costs[-1]))
    chosen = [state]
    for frame_index in range(len(frames) - 1, 0, -1):
        state = int(parent[frame_index, state])
        if state < 0:
            state = chosen[-1]
        chosen.append(state)
    chosen.reverse()
    output = [
        row
        for frame, state in zip(frames, chosen, strict=True)
        for row in states[state].get(frame, [])
    ]
    switches = sum(left != right for left, right in zip(chosen, chosen[1:]))
    return output, {
        "forward": forward_diagnostics,
        "backward": backward_diagnostics,
        "state_switches": switches,
        "forward_frames_selected": chosen.count(0),
        "backward_frames_selected": chosen.count(1),
    }


def refine_directory(
    input_dir: Path,
    seed_dir: Path,
    output_dir: Path,
    sequence_names: list[str] | None = None,
    config: FormationConfig = FormationConfig(),
    *,
    mode: str = "bidirectional",
    geometry_by_sequence: dict[str, Geometry] | None = None,
) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for file in prediction_files(input_dir):
        if allowed is not None and file.stem not in allowed:
            continue
        seeds = read_rows(seed_dir / file.name)
        rows = read_rows(file)
        geometry = (geometry_by_sequence or {}).get(file.stem)
        if mode == "forward":
            refined, diagnostics = refine_sequence(
                rows,
                seeds,
                config,
                direction=1,
                geometry=geometry,
            )
        elif mode == "backward":
            refined, diagnostics = refine_sequence(
                rows,
                seeds,
                config,
                direction=-1,
                geometry=geometry,
            )
        elif mode == "bidirectional":
            refined, diagnostics = bidirectional_refine_sequence(
                rows,
                seeds,
                config,
                geometry,
            )
        else:
            raise ValueError(f"unknown formation mode: {mode}")
        write_rows(output_dir / file.name, refined)
        summary[file.stem] = diagnostics
    return {"sequences": summary, "config": config.__dict__, "mode": mode}


def _names(path: Path | None) -> list[str] | None:
    return None if path is None else [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument(
        "--mode",
        choices=("forward", "backward", "bidirectional"),
        default="bidirectional",
    )
    parser.add_argument("--formation-weight", type=float, default=0.45)
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = refine_directory(
        args.input_dir,
        args.seed_dir,
        args.output_dir,
        _names(args.sequence_list),
        FormationConfig(formation_weight=args.formation_weight),
        mode=args.mode,
    )
    if args.summary_json:
        write_json(args.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
