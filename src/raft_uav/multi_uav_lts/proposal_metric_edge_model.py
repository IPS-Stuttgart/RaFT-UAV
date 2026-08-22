"""Fit LTS edge heads for identity, HOTA@0.05, and CLEAR/IDF1@0.5 utility."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import _proposal_graph_core as core
from ._proposal_edge_likelihood import (
    EDGE_FEATURE_NAMES,
    build_edge_feature_context,
    edge_feature_vector,
    fit_edge_likelihood,
)
from ._proposal_metric_edge_likelihood import (
    MetricEdgeLikelihoodModel,
    write_metric_edge_likelihood_model,
)
from ._records import Detection, box_iou, parse_detection_text, prediction_texts, reject_duplicate_keys
from .proposal_edge_model import _match_proposals_to_truth, _selected_truth_paths
from .proposal_graph_tracker import _parameters, _proposal_rows


@dataclass(frozen=True)
class MetricEdgeTrainingExample:
    sequence: str
    identity_label: int
    hota_005_label: int
    clear_050_label: int
    right_truth_iou: float
    raw_cost: float
    features: tuple[float, ...]


@dataclass(frozen=True)
class MetricEdgeFitResult:
    model: MetricEdgeLikelihoodModel
    examples: tuple[MetricEdgeTrainingExample, ...]
    selected_sequences: tuple[str, ...]
    retained_proposal_rows: int
    truth_matched_proposal_rows: int


def fit_metric_edge_model_from_lts(
    proposal_path: Path,
    truth_dir: Path,
    *,
    sequences: Iterable[str] | None = None,
    min_proposal_confidence: float = 0.003,
    duplicate_iou: float = 0.95,
    min_truth_iou: float = 0.05,
    max_gap: int = 0,
    max_link_cost: float = 2.25,
    negative_candidates_per_left: int = 5,
    enable_common_motion: bool = True,
    swarm_neighbors: int = 4,
    swarm_radius_scale: float = 12.0,
    swarm_unmatched_penalty: float = 2.0,
    l2_penalty: float = 1.0,
    max_iterations: int = 500,
    identity_weight: float = 0.75,
    hota_weight: float = 1.0,
    clear_weight: float = 0.25,
) -> MetricEdgeFitResult:
    """Fit three sequence-balanced calibrated heads on training-only truth."""

    if not 0.0 <= min_truth_iou <= 1.0:
        raise ValueError("min_truth_iou must be in [0, 1]")
    if min_truth_iou > 0.05:
        raise ValueError("min_truth_iou must be <= 0.05 to train the HOTA@0.05 head")
    if negative_candidates_per_left <= 0:
        raise ValueError("negative_candidates_per_left must be positive")
    parameters = _parameters(
        min_proposal_confidence=min_proposal_confidence,
        duplicate_iou=duplicate_iou,
        min_seed_iou=0.05,
        anchor_max_cost=1.25,
        anchor_min_margin=0.15,
        enable_global_links=True,
        max_link_gap=max_gap,
        max_link_cost=max_link_cost,
        center_weight=1.0,
        size_weight=0.25,
        iou_weight=0.35,
        velocity_weight=0.5,
        gap_weight=0.04,
        confidence_weight=0.05,
        enable_common_motion=enable_common_motion,
        common_motion_min_pairs=4,
        common_motion_max_normalized_step=8.0,
        common_motion_max_normalized_residual=1.5,
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
    missing = sorted(expected_names - set(proposal_text))
    if missing:
        raise ValueError("proposal input is missing training sequences: " + ", ".join(missing))

    examples: list[MetricEdgeTrainingExample] = []
    retained_total = 0
    matched_total = 0
    for truth_path in selected_paths:
        sequence = truth_path.stem
        truth_rows = tuple(
            parse_detection_text(truth_path.read_text(encoding="utf-8"), source=str(truth_path))
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
            min_iou=min_truth_iou,
        )
        matched_total += len(truth_mapping)
        truth_lookup = {(row.frame_id, row.object_id): row for row in truth_rows}
        examples.extend(
            _sequence_examples(
                sequence,
                retained,
                truth_mapping,
                truth_lookup,
                parameters,
                negative_candidates_per_left=negative_candidates_per_left,
                swarm_neighbors=swarm_neighbors,
                swarm_radius_scale=swarm_radius_scale,
                swarm_unmatched_penalty=swarm_unmatched_penalty,
            )
        )

    if not examples:
        raise ValueError("no metric-edge training examples were produced")
    features = [example.features for example in examples]
    sequence_ids = [example.sequence for example in examples]
    metadata = {
        "proposal_path": str(proposal_path),
        "truth_dir": str(truth_dir),
        "selected_sequences": [path.stem for path in selected_paths],
        "min_truth_iou": min_truth_iou,
        "max_gap": max_gap,
        "max_link_cost": max_link_cost,
        "negative_candidates_per_left": negative_candidates_per_left,
        "swarm_neighbors": swarm_neighbors,
        "swarm_radius_scale": swarm_radius_scale,
        "swarm_unmatched_penalty": swarm_unmatched_penalty,
        "retained_proposal_rows": retained_total,
        "truth_matched_proposal_rows": matched_total,
    }
    identity = _fit_head(
        features,
        [example.identity_label for example in examples],
        sequence_ids,
        head="identity",
        metadata=metadata,
        l2_penalty=l2_penalty,
        max_iterations=max_iterations,
    )
    hota = _fit_head(
        features,
        [example.hota_005_label for example in examples],
        sequence_ids,
        head="hota_005",
        metadata=metadata,
        l2_penalty=l2_penalty,
        max_iterations=max_iterations,
    )
    clear = _fit_head(
        features,
        [example.clear_050_label for example in examples],
        sequence_ids,
        head="clear_050",
        metadata=metadata,
        l2_penalty=l2_penalty,
        max_iterations=max_iterations,
    )
    model = MetricEdgeLikelihoodModel(
        schema="raft-uav-multi-uav-lts-metric-edge-likelihood-v1",
        identity=identity,
        hota_005=hota,
        clear_050=clear,
        identity_weight=identity_weight,
        hota_weight=hota_weight,
        clear_weight=clear_weight,
        metadata=metadata,
    )
    return MetricEdgeFitResult(
        model=model,
        examples=tuple(examples),
        selected_sequences=tuple(path.stem for path in selected_paths),
        retained_proposal_rows=retained_total,
        truth_matched_proposal_rows=matched_total,
    )


def _fit_head(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    sequence_ids: Sequence[str],
    *,
    head: str,
    metadata: dict[str, object],
    l2_penalty: float,
    max_iterations: int,
):
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives <= 0 or negatives <= 0:
        raise ValueError(f"metric-edge head {head} must contain both classes")
    return fit_edge_likelihood(
        features,
        labels,
        sequence_ids=sequence_ids,
        l2_penalty=l2_penalty,
        max_iterations=max_iterations,
        metadata={**metadata, "metric_head": head},
    )


def _sequence_examples(
    sequence: str,
    retained: tuple[Detection, ...],
    truth_mapping: dict[Detection, int],
    truth_lookup: dict[tuple[int, int], Detection],
    parameters,
    *,
    negative_candidates_per_left: int,
    swarm_neighbors: int,
    swarm_radius_scale: float,
    swarm_unmatched_penalty: float,
) -> list[MetricEdgeTrainingExample]:
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
    examples: list[MetricEdgeTrainingExample] = []
    for left in tracklets:
        left_truth = truth_mapping.get(left.rows[-1])
        if left_truth is None:
            continue
        lower = bisect_left(frames, left.end + 1)
        upper = bisect_right(frames, left.end + parameters.max_link_gap + 1)
        for frame in frames[lower:upper]:
            positive: list[MetricEdgeTrainingExample] = []
            negative: list[MetricEdgeTrainingExample] = []
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
                same_identity = int(left_truth == right_truth)
                truth_row = truth_lookup[(right.rows[0].frame_id, right_truth)]
                overlap = box_iou(right.rows[0], truth_row)
                example = MetricEdgeTrainingExample(
                    sequence=sequence,
                    identity_label=same_identity,
                    hota_005_label=int(same_identity and overlap >= 0.05),
                    clear_050_label=int(same_identity and overlap >= 0.5),
                    right_truth_iou=float(overlap),
                    raw_cost=float(raw_cost),
                    features=tuple(float(value) for value in features),
                )
                (positive if same_identity else negative).append(example)
            examples.extend(sorted(positive, key=_example_sort_key))
            examples.extend(
                sorted(negative, key=_example_sort_key)[:negative_candidates_per_left]
            )
    return examples


def _example_sort_key(example: MetricEdgeTrainingExample) -> tuple[float, tuple[float, ...]]:
    return example.raw_cost, example.features


def write_examples_csv(examples: Sequence[MetricEdgeTrainingExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "sequence",
                "identity_label",
                "hota_005_label",
                "clear_050_label",
                "right_truth_iou",
                "raw_cost",
                *EDGE_FEATURE_NAMES,
            )
        )
        for example in examples:
            writer.writerow(
                (
                    example.sequence,
                    example.identity_label,
                    example.hota_005_label,
                    example.clear_050_label,
                    example.right_truth_iou,
                    example.raw_cost,
                    *example.features,
                )
            )


def write_fit_summary(result: MetricEdgeFitResult, path: Path) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-metric-edge-fit-summary-v1",
        "selected_sequences": list(result.selected_sequences),
        "selected_sequence_count": len(result.selected_sequences),
        "retained_proposal_rows": result.retained_proposal_rows,
        "truth_matched_proposal_rows": result.truth_matched_proposal_rows,
        "training_examples": len(result.examples),
        "identity_positive_edges": sum(example.identity_label for example in result.examples),
        "hota_005_positive_edges": sum(example.hota_005_label for example in result.examples),
        "clear_050_positive_edges": sum(example.clear_050_label for example in result.examples),
        "model": result.model.to_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--min-truth-iou", type=float, default=0.05)
    parser.add_argument("--max-gap", type=int, default=0)
    parser.add_argument("--max-link-cost", type=float, default=2.25)
    parser.add_argument("--negative-candidates-per-left", type=int, default=5)
    parser.add_argument("--disable-common-motion", action="store_true")
    parser.add_argument("--swarm-neighbors", type=int, default=4)
    parser.add_argument("--swarm-radius-scale", type=float, default=12.0)
    parser.add_argument("--swarm-unmatched-penalty", type=float, default=2.0)
    parser.add_argument("--l2-penalty", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--identity-weight", type=float, default=0.75)
    parser.add_argument("--hota-weight", type=float, default=1.0)
    parser.add_argument("--clear-weight", type=float, default=0.25)
    args = parser.parse_args(argv)
    result = fit_metric_edge_model_from_lts(
        args.proposal_path,
        args.truth_dir,
        sequences=args.sequences,
        min_proposal_confidence=args.min_proposal_confidence,
        duplicate_iou=args.duplicate_iou,
        min_truth_iou=args.min_truth_iou,
        max_gap=args.max_gap,
        max_link_cost=args.max_link_cost,
        negative_candidates_per_left=args.negative_candidates_per_left,
        enable_common_motion=not args.disable_common_motion,
        swarm_neighbors=args.swarm_neighbors,
        swarm_radius_scale=args.swarm_radius_scale,
        swarm_unmatched_penalty=args.swarm_unmatched_penalty,
        l2_penalty=args.l2_penalty,
        max_iterations=args.max_iterations,
        identity_weight=args.identity_weight,
        hota_weight=args.hota_weight,
        clear_weight=args.clear_weight,
    )
    write_metric_edge_likelihood_model(result.model, args.output_json)
    if args.summary_json is not None:
        write_fit_summary(result, args.summary_json)
    if args.examples_csv is not None:
        write_examples_csv(result.examples, args.examples_csv)
    print(json.dumps(asdict(result.model), default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
