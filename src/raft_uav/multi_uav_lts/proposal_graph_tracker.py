"""Offline variable-cardinality tracking for Multi-UAV LTS proposal banks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ._proposal_graph_core import track_sequence
from ._records import (
    Detection,
    format_detection,
    parse_detection_text,
    prediction_texts,
    reject_duplicate_keys,
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)


@dataclass(frozen=True)
class ProposalGraphParameters:
    min_proposal_confidence: float
    duplicate_iou: float
    min_seed_iou: float
    anchor_max_cost: float
    anchor_min_margin: float
    enable_global_links: bool
    max_link_gap: int
    max_link_cost: float
    center_weight: float
    size_weight: float
    iou_weight: float
    velocity_weight: float
    gap_weight: float
    confidence_weight: float
    birth_min_hits: int
    birth_min_span: int
    birth_min_mean_confidence: float
    image_width: float | None
    image_height: float | None
    border_margin_fraction: float
    border_gap_discount: float


@dataclass(frozen=True)
class SequenceProposalGraphSummary:
    sequence: str
    seed_count: int
    input_proposal_rows: int
    retained_proposal_rows: int
    duplicate_suppressed_rows: int
    anchor_tracklets: int
    graph_links: int
    seeded_paths: int
    confirmed_birth_paths: int
    dropped_unseeded_paths: int
    output_rows: int
    output_ids: int


@dataclass(frozen=True)
class ProposalGraphSummary:
    schema: str
    proposal_path: str
    first_frame_label_dir: str
    output_dir: str
    parameters: ProposalGraphParameters
    sequence_count: int
    seed_count: int
    input_proposal_rows: int
    retained_proposal_rows: int
    duplicate_suppressed_rows: int
    anchor_tracklets: int
    graph_links: int
    seeded_paths: int
    confirmed_birth_paths: int
    dropped_unseeded_paths: int
    output_rows: int
    output_ids: int
    sequences: tuple[SequenceProposalGraphSummary, ...]


def track_proposal_graph(
    proposal_path: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *,
    min_proposal_confidence: float = 0.003,
    duplicate_iou: float = 0.95,
    min_seed_iou: float = 0.05,
    anchor_max_cost: float = 1.25,
    anchor_min_margin: float = 0.15,
    enable_global_links: bool = True,
    max_link_gap: int = 30,
    max_link_cost: float = 2.25,
    center_weight: float = 1.0,
    size_weight: float = 0.25,
    iou_weight: float = 0.35,
    velocity_weight: float = 0.5,
    gap_weight: float = 0.04,
    confidence_weight: float = 0.05,
    birth_min_hits: int = 3,
    birth_min_span: int = 2,
    birth_min_mean_confidence: float = 0.003,
    image_width: float | None = None,
    image_height: float | None = None,
    border_margin_fraction: float = 0.08,
    border_gap_discount: float = 0.35,
    sequences: Iterable[str] | None = None,
) -> ProposalGraphSummary:
    parameters = _parameters(
        min_proposal_confidence=min_proposal_confidence,
        duplicate_iou=duplicate_iou,
        min_seed_iou=min_seed_iou,
        anchor_max_cost=anchor_max_cost,
        anchor_min_margin=anchor_min_margin,
        enable_global_links=enable_global_links,
        max_link_gap=max_link_gap,
        max_link_cost=max_link_cost,
        center_weight=center_weight,
        size_weight=size_weight,
        iou_weight=iou_weight,
        velocity_weight=velocity_weight,
        gap_weight=gap_weight,
        confidence_weight=confidence_weight,
        birth_min_hits=birth_min_hits,
        birth_min_span=birth_min_span,
        birth_min_mean_confidence=birth_min_mean_confidence,
        image_width=image_width,
        image_height=image_height,
        border_margin_fraction=border_margin_fraction,
        border_gap_discount=border_gap_discount,
    )
    label_paths = _input_paths(proposal_path, first_frame_label_dir, output_dir)
    proposal_text = prediction_texts(proposal_path)
    expected = {f"{path.stem}.txt" for path in label_paths}
    unexpected = sorted(set(proposal_text) - expected)
    if unexpected:
        raise ValueError(
            "proposal input contains unknown sequence files: " + ", ".join(unexpected)
        )
    requested = set(sequences or ())
    available = {path.stem for path in label_paths}
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.txt"):
        stale.unlink()
    summaries: list[SequenceProposalGraphSummary] = []
    for label_path in label_paths:
        sequence = label_path.stem
        if requested and sequence not in requested:
            continue
        seeds = _seed_rows(label_path)
        proposals = tuple(
            parse_detection_text(
                proposal_text.get(f"{sequence}.txt", ""),
                source=f"{proposal_path}:{sequence}.txt",
            )
        )
        result = track_sequence(seeds, proposals, parameters)
        (output_dir / f"{sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in result.rows),
            encoding="utf-8",
        )
        summaries.append(
            SequenceProposalGraphSummary(
                sequence=sequence,
                seed_count=len(seeds),
                input_proposal_rows=len(proposals),
                retained_proposal_rows=result.retained_rows,
                duplicate_suppressed_rows=result.suppressed_rows,
                anchor_tracklets=result.anchor_tracklets,
                graph_links=result.graph_links,
                seeded_paths=result.seeded_paths,
                confirmed_birth_paths=result.birth_paths,
                dropped_unseeded_paths=result.dropped_paths,
                output_rows=len(result.rows),
                output_ids=len({row.object_id for row in result.rows}),
            )
        )
    return _summary(
        proposal_path,
        first_frame_label_dir,
        output_dir,
        parameters,
        tuple(summaries),
    )


def _parameters(**raw: object) -> ProposalGraphParameters:
    unit = lambda key: validate_unit_interval(raw[key], name=key)
    finite = lambda key: validate_nonnegative_finite(raw[key], name=key)
    positive = lambda key: _positive(raw[key], name=key)
    integer = lambda key: validate_nonnegative_int(raw[key], name=key)
    global_links = raw["enable_global_links"]
    if not isinstance(global_links, bool):
        raise ValueError("enable_global_links must be a Boolean")
    birth_hits = integer("birth_min_hits")
    if birth_hits <= 0:
        raise ValueError("birth_min_hits must be a positive integer")
    weights = (
        finite("center_weight"),
        finite("size_weight"),
        finite("iou_weight"),
    )
    if sum(weights) <= 0.0:
        raise ValueError("at least one geometric association weight must be positive")
    width = _optional_positive(raw["image_width"], name="image_width")
    height = _optional_positive(raw["image_height"], name="image_height")
    if (width is None) != (height is None):
        raise ValueError("image_width and image_height must be supplied together")
    return ProposalGraphParameters(
        min_proposal_confidence=unit("min_proposal_confidence"),
        duplicate_iou=unit("duplicate_iou"),
        min_seed_iou=unit("min_seed_iou"),
        anchor_max_cost=positive("anchor_max_cost"),
        anchor_min_margin=finite("anchor_min_margin"),
        enable_global_links=global_links,
        max_link_gap=integer("max_link_gap"),
        max_link_cost=positive("max_link_cost"),
        center_weight=weights[0],
        size_weight=weights[1],
        iou_weight=weights[2],
        velocity_weight=finite("velocity_weight"),
        gap_weight=finite("gap_weight"),
        confidence_weight=finite("confidence_weight"),
        birth_min_hits=birth_hits,
        birth_min_span=integer("birth_min_span"),
        birth_min_mean_confidence=unit("birth_min_mean_confidence"),
        image_width=width,
        image_height=height,
        border_margin_fraction=unit("border_margin_fraction"),
        border_gap_discount=unit("border_gap_discount"),
    )


def _positive(value: object, *, name: str) -> float:
    parsed = validate_nonnegative_finite(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _optional_positive(value: object, *, name: str) -> float | None:
    return None if value is None else _positive(value, name=name)


def _input_paths(
    proposal_path: Path,
    label_dir: Path,
    output_dir: Path,
) -> list[Path]:
    if not proposal_path.exists():
        raise FileNotFoundError(f"proposal input does not exist: {proposal_path}")
    if not label_dir.exists():
        raise FileNotFoundError(f"first-frame label directory does not exist: {label_dir}")
    if not label_dir.is_dir():
        raise NotADirectoryError(label_dir)
    labels = sorted(label_dir.glob("*.txt"))
    if not labels:
        raise ValueError(f"first-frame label directory contains no .txt files: {label_dir}")
    output = output_dir.resolve()
    label_root = label_dir.resolve()
    if output == label_root or label_root in output.parents:
        raise ValueError("output directory must not alias or be nested in seed labels")
    if proposal_path.is_dir():
        proposal_root = proposal_path.resolve()
        if output == proposal_root or proposal_root in output.parents:
            raise ValueError("output directory must not alias or be nested in proposals")
    return labels


def _seed_rows(path: Path) -> tuple[Detection, ...]:
    rows = parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))
    if any(row.frame_id != 1 for row in rows):
        raise ValueError(f"{path}: expected first-frame-only labels")
    reject_duplicate_keys(rows, label="seed")
    return tuple(sorted(rows, key=lambda row: row.object_id))


def _summary(
    proposal_path: Path,
    label_dir: Path,
    output_dir: Path,
    parameters: ProposalGraphParameters,
    rows: tuple[SequenceProposalGraphSummary, ...],
) -> ProposalGraphSummary:
    total = lambda name: sum(getattr(row, name) for row in rows)
    return ProposalGraphSummary(
        schema="raft-uav-multi-uav-lts-proposal-graph-v1",
        proposal_path=str(proposal_path),
        first_frame_label_dir=str(label_dir),
        output_dir=str(output_dir),
        parameters=parameters,
        sequence_count=len(rows),
        seed_count=total("seed_count"),
        input_proposal_rows=total("input_proposal_rows"),
        retained_proposal_rows=total("retained_proposal_rows"),
        duplicate_suppressed_rows=total("duplicate_suppressed_rows"),
        anchor_tracklets=total("anchor_tracklets"),
        graph_links=total("graph_links"),
        seeded_paths=total("seeded_paths"),
        confirmed_birth_paths=total("confirmed_birth_paths"),
        dropped_unseeded_paths=total("dropped_unseeded_paths"),
        output_rows=total("output_rows"),
        output_ids=total("output_ids"),
        sequences=rows,
    )


def write_summary(summary: ProposalGraphSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_path", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--min-proposal-confidence", type=float, default=0.003)
    parser.add_argument("--duplicate-iou", type=float, default=0.95)
    parser.add_argument("--min-seed-iou", type=float, default=0.05)
    parser.add_argument("--anchor-max-cost", type=float, default=1.25)
    parser.add_argument("--anchor-min-margin", type=float, default=0.15)
    parser.add_argument("--no-global-links", action="store_true")
    parser.add_argument("--max-link-gap", type=int, default=30)
    parser.add_argument("--max-link-cost", type=float, default=2.25)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.25)
    parser.add_argument("--iou-weight", type=float, default=0.35)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--gap-weight", type=float, default=0.04)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument("--birth-min-hits", type=int, default=3)
    parser.add_argument("--birth-min-span", type=int, default=2)
    parser.add_argument("--birth-min-mean-confidence", type=float, default=0.003)
    parser.add_argument("--image-width", type=float)
    parser.add_argument("--image-height", type=float)
    parser.add_argument("--border-margin-fraction", type=float, default=0.08)
    parser.add_argument("--border-gap-discount", type=float, default=0.35)
    args = parser.parse_args(argv)
    keywords = vars(args).copy()
    proposal_path = keywords.pop("proposal_path")
    label_dir = keywords.pop("first_frame_label_dir")
    output_dir = keywords.pop("output_dir")
    output_json = keywords.pop("output_json")
    keywords["enable_global_links"] = not keywords.pop("no_global_links")
    summary = track_proposal_graph(proposal_path, label_dir, output_dir, **keywords)
    if output_json:
        write_summary(summary, output_json)
    print(f"sequence_count={summary.sequence_count}")
    print(f"output_rows={summary.output_rows}")
    print(f"confirmed_birth_paths={summary.confirmed_birth_paths}")
    print(f"dropped_unseeded_paths={summary.dropped_unseeded_paths}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
