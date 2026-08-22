"""Fit a calibrated same-identity edge model from Multi-UAV LTS training data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import _proposal_graph_core as core
from ._proposal_edge_likelihood import (
    EDGE_FEATURE_NAMES,
    EdgeLikelihoodModel,
    build_edge_feature_context,
    edge_feature_vector,
    fit_edge_likelihood,
    write_edge_likelihood_model,
)
from ._records import (
    Detection,
    box_iou,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
)
from .proposal_graph_tracker import _parameters, _proposal_rows


@dataclass(frozen=True)
class EdgeTrainingExample:
    sequence: str
    label: int
    raw_cost: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class EdgeModelFitResult:
    model: EdgeLikelihoodModel
    examples: tuple[EdgeTrainingExample, ...]
    selected_sequences: tuple[str, ...]
    retained_proposal_rows: int
    truth_matched_proposal_rows: int
    positive_candidate_edges: int
    negative_candidate_edges: int


def fit_edge_model_from_lts(
    proposal_path: Path,
    truth_dir: Path,
    *,
    sequences: Iterable[str] | None = None,
    min_proposal_confidence: float = 0.003,
    duplicate_iou: float = 0.95,
    min_truth_iou: float = 0.3,
    max_gap: int = 0,
    max_link_cost: float = 2.25,
    negative_candidates_per_left: int = 5,
    center_weight: float = 1.0,
    size_weight: float = 0.25,
    iou_weight: float = 0.35,
    velocity_weight: float = 0.5,
    gap_weight: float = 0.04,
    confidence_weight: float = 0.05,
    enable_common_motion: bool = False,
    common_motion_min_pairs: int = 4,
    common_motion_max_normalized_step: float = 8.0,
    common_motion_max_normalized_residual: float = 1.5,
    swarm_neighbors: int = 4,
    swarm_radius_scale: float = 12.0,
    swarm_unmatched_penalty: float = 2.0,
    l2_penalty: float = 1.0,
    max_iterations: int = 500,
) -> EdgeModelFitResult:
    """Fit the edge model using truth only for training-side edge labels."""

    truth_iou = _unit_interval(min_truth_iou, name="min_truth_iou")
    negative_limit = _positive_int(
        negative_candidates_per_left,
        name="negative_candidates_per_left",
    )
    neighbors = _positive_int(swarm_neighbors, name="swarm_neighbors")
    radius_scale = _positive_finite(
        swarm_radius_scale,
        name="swarm_radius_scale",
    )
    unmatched_penalty = _positive_finite(
        swarm_unmatched_penalty,
        name="swarm_unmatched_penalty",
    )
    penalty = _nonnegative_finite(l2_penalty, name="l2_penalty")
    iterations = _positive_int(max_iterations, name="max_iterations")
    parameters = _parameters(
        min_proposal_confidence=min_proposal_confidence,
        duplicate_iou=duplicate_iou,
        min_seed_iou=0.05,
        anchor_max_cost=1.25,
        anchor_min_margin=0.15,
        enable_global_links=True,
        max_link_gap=max_gap,
        max_link_cost=max_link_cost,
        center_weight=center_weight,
        size_weight=size_weight,
        iou_weight=iou_weight,
        velocity_weight=velocity_weight,
        gap_weight=gap_weight,
        confidence_weight=confidence_weight,
        enable_common_motion=enable_common_motion,
        common_motion_min_pairs=common_motion_min_pairs,
        common_motion_max_normalized_step=common_motion_max_normalized_step,
        common_motion_max_normalized_residual=common_motion_max_normalized_residual,
        interpolate_max_gap=0,
        birth_min_hits=3,
        birth_min_span=2,
        birth_min_mean_confidence=min_proposal_confidence,
        birth_require_border_entry=False,
        birth_min_inward_motion=0.0,
        image_width=None,
        image_height=None,
        border_margin_fraction=0.08,
        border_gap_discount=0.35,
    )
    selected_paths = _selected_truth_paths(truth_dir, sequences)
    proposal_text = prediction_texts(proposal_path)
    expected_names = {f"{path.stem}.txt" for path in selected_paths}
    missing_proposals = sorted(expected_names - set(proposal_text))
    if missing_proposals:
        raise ValueError(
            "proposal input is missing training sequences: "
            + ", ".join(missing_proposals)
        )

    examples: list[EdgeTrainingExample] = []
    retained_total = 0
    matched_total = 0
    for truth_path in selected_paths:
        sequence = truth_path.stem
        truth_rows = tuple(
            parse_detection_text(
                truth_path.read_text(encoding="utf-8"),
                source=str(truth_path),
            )
        )
        reject_duplicate_keys(list(truth_rows), label="truth")
        proposals = _proposal_rows(
            proposal_text[f"{sequence}.txt"],
            source=f"{proposal_path}:{sequence}.txt",
        )
        retained, _suppressed = core._canonicalize(proposals, parameters)
        retained_total += len(retained)
        truth_mapping = _match_proposals_to_truth(
            retained,
            truth_rows,
            min_iou=truth_iou,
        )
        matched_total += len(truth_mapping)
        examples.extend(
            _sequence_examples(
                sequence,
                retained,
                truth_mapping,
                parameters,
                negative_candidates_per_left=negative_limit,
                swarm_neighbors=neighbors,
                swarm_radius_scale=radius_scale,
                swarm_unmatched_penalty=unmatched_penalty,
            )
        )

    if not examples:
        raise ValueError("no labelled candidate edges were produced")
    feature_matrix = [example.features for example in examples]
    labels = [example.label for example in examples]
    sequence_ids = [example.sequence for example in examples]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    metadata = {
        "proposal_path": str(proposal_path),
        "truth_dir": str(truth_dir),
        "selected_sequences": [path.stem for path in selected_paths],
        "min_proposal_confidence": parameters.min_proposal_confidence,
        "duplicate_iou": parameters.duplicate_iou,
        "min_truth_iou": truth_iou,
        "max_gap": parameters.max_link_gap,
        "max_link_cost": parameters.max_link_cost,
        "negative_candidates_per_left": negative_limit,
        "enable_common_motion": parameters.enable_common_motion,
        "common_motion_min_pairs": parameters.common_motion_min_pairs,
        "common_motion_max_normalized_step": (
            parameters.common_motion_max_normalized_step
        ),
        "common_motion_max_normalized_residual": (
            parameters.common_motion_max_normalized_residual
        ),
        "swarm_neighbors": neighbors,
        "swarm_radius_scale": radius_scale,
        "swarm_unmatched_penalty": unmatched_penalty,
        "retained_proposal_rows": retained_total,
        "truth_matched_proposal_rows": matched_total,
    }
    model = fit_edge_likelihood(
        feature_matrix,
        labels,
        sequence_ids=sequence_ids,
        l2_penalty=penalty,
        max_iterations=iterations,
        metadata=metadata,
    )
    return EdgeModelFitResult(
        model=model,
        examples=tuple(examples),
        selected_sequences=tuple(path.stem for path in selected_paths),
        retained_proposal_rows=retained_total,
        truth_matched_proposal_rows=matched_total,
        positive_candidate_edges=positive_count,
        negative_candidate_edges=negative_count,
    )


def _sequence_examples(
    sequence: str,
    retained: tuple[Detection, ...],
    truth_mapping: dict[Detection, int],
    parameters,
    *,
    negative_candidates_per_left: int,
    swarm_neighbors: int,
    swarm_radius_scale: float,
    swarm_unmatched_penalty: float,
) -> list[EdgeTrainingExample]:
    nodes = tuple(core._Node(index, row) for index, row in enumerate(retained))
    common_motion = core._estimate_common_motion(nodes, parameters)
    tracklets = tuple(
        core._Tracklet(index, (node.row,), (node.index,))
        for index, node in enumerate(nodes)
    )
    starts: dict[int, list[core._Tracklet]] = {}
    for tracklet in tracklets:
        starts.setdefault(tracklet.start, []).append(tracklet)
    frames = sorted(starts)
    context = build_edge_feature_context(
        retained,
        neighbor_count=swarm_neighbors,
        radius_scale=swarm_radius_scale,
        unmatched_penalty=swarm_unmatched_penalty,
    )
    examples: list[EdgeTrainingExample] = []
    for left in tracklets:
        left_truth = truth_mapping.get(left.rows[-1])
        if left_truth is None:
            continue
        lower = bisect_left(frames, left.end + 1)
        upper = bisect_right(frames, left.end + parameters.max_link_gap + 1)
        for frame in frames[lower:upper]:
            positive: list[EdgeTrainingExample] = []
            negative: list[EdgeTrainingExample] = []
            for right in starts[frame]:
                right_truth = truth_mapping.get(right.rows[0])
                if right_truth is None:
                    continue
                raw_cost = core._link_cost(left, right, parameters, common_motion)
                if not math.isfinite(raw_cost) or raw_cost >= parameters.max_link_cost:
                    continue
                predicted = core._predict(left.rows, right.start, common_motion)
                features = edge_feature_vector(
                    left.rows[-1],
                    right.rows[0],
                    predicted,
                    gap_frames=right.start - left.end - 1,
                    context=context,
                )
                example = EdgeTrainingExample(
                    sequence=sequence,
                    label=int(left_truth == right_truth),
                    raw_cost=raw_cost,
                    features=tuple(float(value) for value in features),
                )
                (positive if example.label else negative).append(example)
            examples.extend(sorted(positive, key=_example_sort_key))
            examples.extend(
                sorted(negative, key=_example_sort_key)[:negative_candidates_per_left]
            )
    return examples


def _example_sort_key(example: EdgeTrainingExample) -> tuple[float, tuple[float, ...]]:
    return example.raw_cost, example.features


def _match_proposals_to_truth(
    proposals: Sequence[Detection],
    truth: Sequence[Detection],
    *,
    min_iou: float,
) -> dict[Detection, int]:
    proposal_frames: dict[int, list[Detection]] = {}
    truth_frames: dict[int, list[Detection]] = {}
    for row in proposals:
        proposal_frames.setdefault(row.frame_id, []).append(row)
    for row in truth:
        truth_frames.setdefault(row.frame_id, []).append(row)

    mapping: dict[Detection, int] = {}
    for frame in sorted(set(proposal_frames) & set(truth_frames)):
        proposal_rows = proposal_frames[frame]
        truth_rows = truth_frames[frame]
        overlaps = np.asarray(
            [
                [box_iou(truth_row, proposal_row) for proposal_row in proposal_rows]
                for truth_row in truth_rows
            ],
            dtype=float,
        )
        valid = (overlaps > 0.0) & (overlaps >= min_iou)
        bonus = float(min(overlaps.shape) + 1)
        rows, columns = linear_sum_assignment(
            -np.where(valid, bonus + overlaps, 0.0)
        )
        for truth_index, proposal_index in zip(rows, columns, strict=True):
            if valid[int(truth_index), int(proposal_index)]:
                mapping[proposal_rows[int(proposal_index)]] = truth_rows[
                    int(truth_index)
                ].object_id
    return mapping


def _selected_truth_paths(
    truth_dir: Path,
    sequences: Iterable[str] | None,
) -> tuple[Path, ...]:
    if not truth_dir.is_dir():
        raise NotADirectoryError(truth_dir)
    available = {path.stem: path for path in sorted(truth_dir.glob("*.txt"))}
    if not available:
        raise ValueError(f"truth directory contains no .txt files: {truth_dir}")
    requested = tuple(dict.fromkeys(str(value) for value in (sequences or ())))
    if not requested:
        return tuple(available.values())
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"unknown truth sequences: {', '.join(missing)}")
    return tuple(available[name] for name in requested)


def write_examples_csv(examples: Sequence[EdgeTrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sequence", "label", "raw_cost", *EDGE_FEATURE_NAMES))
        for example in examples:
            writer.writerow(
                (example.sequence, example.label, example.raw_cost, *example.features)
            )


def write_fit_summary(result: EdgeModelFitResult, path: Path) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-edge-model-fit-summary-v1",
        "selected_sequences": list(result.selected_sequences),
        "selected_sequence_count": len(result.selected_sequences),
        "retained_proposal_rows": result.retained_proposal_rows,
        "truth_matched_proposal_rows": result.truth_matched_proposal_rows,
        "training_examples": len(result.examples),
        "positive_candidate_edges": result.positive_candidate_edges,
        "negative_candidate_edges": result.negative_candidate_edges,
        "model": result.model.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_path", type=Path)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--examples-csv", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--min-proposal-confidence", type=float, default=0.003)
    parser.add_argument("--duplicate-iou", type=float, default=0.95)
    parser.add_argument("--min-truth-iou", type=float, default=0.3)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--max-link-cost", type=float, default=2.25)
    parser.add_argument("--negative-candidates-per-left", type=int, default=5)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.25)
    parser.add_argument("--iou-weight", type=float, default=0.35)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--gap-weight", type=float, default=0.04)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--enable-common-motion", action="store_true")
    parser.add_argument("--common-motion-min-pairs", type=int, default=4)
    parser.add_argument(
        "--common-motion-max-normalized-step", type=float, default=8.0
    )
    parser.add_argument(
        "--common-motion-max-normalized-residual", type=float, default=1.5
    )
    parser.add_argument("--swarm-neighbors", type=int, default=4)
    parser.add_argument("--swarm-radius-scale", type=float, default=12.0)
    parser.add_argument("--swarm-unmatched-penalty", type=float, default=2.0)
    parser.add_argument("--l2-penalty", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    args = parser.parse_args(argv)
    result = fit_edge_model_from_lts(
        args.proposal_path,
        args.truth_dir,
        sequences=args.sequences,
        min_proposal_confidence=args.min_proposal_confidence,
        duplicate_iou=args.duplicate_iou,
        min_truth_iou=args.min_truth_iou,
        max_gap=args.max_gap,
        max_link_cost=args.max_link_cost,
        negative_candidates_per_left=args.negative_candidates_per_left,
        center_weight=args.center_weight,
        size_weight=args.size_weight,
        iou_weight=args.iou_weight,
        velocity_weight=args.velocity_weight,
        gap_weight=args.gap_weight,
        confidence_weight=args.confidence_weight,
        enable_common_motion=args.enable_common_motion,
        common_motion_min_pairs=args.common_motion_min_pairs,
        common_motion_max_normalized_step=(
            args.common_motion_max_normalized_step
        ),
        common_motion_max_normalized_residual=(
            args.common_motion_max_normalized_residual
        ),
        swarm_neighbors=args.swarm_neighbors,
        swarm_radius_scale=args.swarm_radius_scale,
        swarm_unmatched_penalty=args.swarm_unmatched_penalty,
        l2_penalty=args.l2_penalty,
        max_iterations=args.max_iterations,
    )
    write_edge_likelihood_model(result.model, args.output_json)
    if args.summary_json:
        write_fit_summary(result, args.summary_json)
    if args.examples_csv:
        write_examples_csv(result.examples, args.examples_csv)
    print(f"selected_sequences={len(result.selected_sequences)}")
    print(f"training_examples={len(result.examples)}")
    print(f"positive_candidate_edges={result.positive_candidate_edges}")
    print(f"negative_candidate_edges={result.negative_candidate_edges}")
    return 0


def _unit_interval(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value



def _positive_finite(value: object, *, name: str) -> float:
    parsed = _nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed

def _nonnegative_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
