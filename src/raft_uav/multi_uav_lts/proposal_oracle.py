"""Audit low-threshold detector proposal banks for Multi-UAV LTS."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import (
    Detection,
    box_iou,
    format_detection,
    parse_detection_text,
    prediction_texts,
    rows_by_frame,
    validate_unit_interval,
)
from .metrics import evaluate_lts_predictions


DEFAULT_CONFIDENCE_THRESHOLDS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.09)
DEFAULT_IOU_THRESHOLDS = (0.05, 0.3, 0.5)
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SIZE_BINS = (
    ("area_lt_64", 0.0, 64.0),
    ("area_64_144", 64.0, 144.0),
    ("area_144_400", 144.0, 400.0),
    ("area_ge_400", 400.0, float("inf")),
)
_EPS = np.finfo(float).eps


@dataclass(frozen=True)
class ProposalSource:
    name: str
    path: str
    fused: bool


@dataclass(frozen=True)
class CoverageRow:
    source: str
    confidence_threshold: float
    iou_threshold: float
    sequence_count: int
    truth_count: int
    proposal_count: int
    matched_count: int
    recall: float
    mean_matched_iou: float
    mean_best_iou: float


@dataclass(frozen=True)
class SequenceCoverageRow:
    source: str
    sequence: str
    confidence_threshold: float
    iou_threshold: float
    truth_count: int
    proposal_count: int
    matched_count: int
    recall: float
    mean_matched_iou: float
    mean_best_iou: float


@dataclass(frozen=True)
class SizeCoverageRow:
    source: str
    confidence_threshold: float
    iou_threshold: float
    size_bin: str
    truth_count: int
    matched_count: int
    recall: float


@dataclass(frozen=True)
class OracleScore:
    source: str
    confidence_threshold: float
    assignment_iou_threshold: float
    exact_seed_rows: bool
    output_dir: str
    output_rows: int
    codabench_hota: float
    codabench_mota: float
    codabench_idf1: float
    hota: float
    deta: float
    assa: float
    loca: float


@dataclass(frozen=True)
class ProposalOracleSummary:
    schema: str
    truth_dir: str
    output_dir: str
    selected_sequences: tuple[str, ...]
    confidence_thresholds: tuple[float, ...]
    iou_thresholds: tuple[float, ...]
    oracle_confidence_threshold: float
    oracle_iou_threshold: float
    sources: tuple[ProposalSource, ...]
    coverage: tuple[CoverageRow, ...]
    sequence_coverage: tuple[SequenceCoverageRow, ...]
    size_coverage: tuple[SizeCoverageRow, ...]
    oracle_scores: tuple[OracleScore, ...]


@dataclass(frozen=True)
class _Match:
    truth_index: int
    proposal_index: int
    iou: float


@dataclass(frozen=True)
class _CoverageCounts:
    truth_count: int
    proposal_count: int
    matched_count: int
    matched_iou_sum: float
    best_iou_sum: float
    size_truth: Mapping[str, int]
    size_matched: Mapping[str, int]


def audit_proposal_banks(
    proposal_paths: Mapping[str, Path],
    truth_dir: Path,
    output_dir: Path,
    *,
    confidence_thresholds: Sequence[float] = DEFAULT_CONFIDENCE_THRESHOLDS,
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    oracle_confidence_threshold: float = 0.003,
    oracle_iou_threshold: float = 0.05,
    include_fused: bool = True,
    sequences: Iterable[str] | None = None,
) -> ProposalOracleSummary:
    """Measure proposal recall and materialize identity-oracle upper bounds."""

    if not proposal_paths:
        raise ValueError("at least one proposal source is required")
    normalized_paths = _normalize_proposal_paths(proposal_paths)
    confidence_values = _validated_thresholds(
        confidence_thresholds,
        name="confidence_thresholds",
    )
    iou_values = _validated_thresholds(iou_thresholds, name="iou_thresholds")
    oracle_confidence = validate_unit_interval(
        oracle_confidence_threshold,
        name="oracle_confidence_threshold",
    )
    oracle_iou = validate_unit_interval(
        oracle_iou_threshold,
        name="oracle_iou_threshold",
    )
    truth_paths = _truth_paths(truth_dir)
    _reject_output_aliases(normalized_paths, truth_dir, output_dir)

    requested = set(sequences or ())
    truth_names = {path.stem for path in truth_paths}
    if requested:
        missing = sorted(requested - truth_names)
        if missing:
            raise ValueError(f"unknown truth sequences: {', '.join(missing)}")
    selected_paths = [path for path in truth_paths if not requested or path.stem in requested]
    selected_sequences = tuple(path.stem for path in selected_paths)

    truth_by_sequence = {
        path.stem: parse_detection_text(
            path.read_text(encoding="utf-8"),
            source=str(path),
        )
        for path in selected_paths
    }
    proposals_by_source = {
        name: _load_source(path, selected_sequences, truth_names)
        for name, path in normalized_paths.items()
    }
    source_rows: dict[str, dict[str, tuple[Detection, ...]]] = {
        name: {
            sequence: tuple(_canonicalize_proposals(rows.get(sequence, ())))
            for sequence in selected_sequences
        }
        for name, rows in proposals_by_source.items()
    }
    sources = [
        ProposalSource(name=name, path=str(path), fused=False)
        for name, path in normalized_paths.items()
    ]
    if include_fused and len(source_rows) > 1:
        fused_name = "fused"
        if fused_name in source_rows:
            raise ValueError("proposal source name 'fused' is reserved")
        source_rows[fused_name] = {
            sequence: tuple(
                _canonicalize_proposals(
                    row
                    for name in normalized_paths
                    for row in source_rows[name][sequence]
                )
            )
            for sequence in selected_sequences
        }
        sources.append(ProposalSource(name=fused_name, path="<union>", fused=True))

    coverage_rows: list[CoverageRow] = []
    sequence_rows: list[SequenceCoverageRow] = []
    size_rows: list[SizeCoverageRow] = []
    for source, by_sequence in source_rows.items():
        for confidence_threshold in confidence_values:
            for iou_threshold in iou_values:
                sequence_counts: list[_CoverageCounts] = []
                for sequence in selected_sequences:
                    counts = _coverage_counts(
                        truth_by_sequence[sequence],
                        by_sequence[sequence],
                        confidence_threshold=confidence_threshold,
                        iou_threshold=iou_threshold,
                    )
                    sequence_counts.append(counts)
                    sequence_rows.append(
                        _sequence_coverage_row(
                            source,
                            sequence,
                            confidence_threshold,
                            iou_threshold,
                            counts,
                        )
                    )
                aggregate = _combine_counts(sequence_counts)
                coverage_rows.append(
                    _coverage_row(
                        source,
                        confidence_threshold,
                        iou_threshold,
                        len(selected_sequences),
                        aggregate,
                    )
                )
                for size_bin, _lower, _upper in _SIZE_BINS:
                    truth_count = aggregate.size_truth[size_bin]
                    matched_count = aggregate.size_matched[size_bin]
                    size_rows.append(
                        SizeCoverageRow(
                            source=source,
                            confidence_threshold=confidence_threshold,
                            iou_threshold=iou_threshold,
                            size_bin=size_bin,
                            truth_count=truth_count,
                            matched_count=matched_count,
                            recall=_ratio(matched_count, truth_count),
                        )
                    )

    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_scores = tuple(
        _materialize_oracle(
            source,
            source_rows[source],
            truth_by_sequence,
            truth_dir,
            output_dir,
            selected_sequences,
            confidence_threshold=oracle_confidence,
            iou_threshold=oracle_iou,
        )
        for source in source_rows
    )
    summary = ProposalOracleSummary(
        schema="raft-uav-multi-uav-lts-proposal-oracle-v1",
        truth_dir=str(truth_dir),
        output_dir=str(output_dir),
        selected_sequences=selected_sequences,
        confidence_thresholds=confidence_values,
        iou_thresholds=iou_values,
        oracle_confidence_threshold=oracle_confidence,
        oracle_iou_threshold=oracle_iou,
        sources=tuple(sources),
        coverage=tuple(coverage_rows),
        sequence_coverage=tuple(sequence_rows),
        size_coverage=tuple(size_rows),
        oracle_scores=oracle_scores,
    )
    _write_artifacts(summary, output_dir)
    return summary


def _normalize_proposal_paths(proposal_paths: Mapping[str, Path]) -> dict[str, Path]:
    normalized: dict[str, Path] = {}
    for raw_name, raw_path in proposal_paths.items():
        name = raw_name.strip()
        if (
            not name
            or name in {".", "..", "fused"}
            or not _NAME_PATTERN.fullmatch(name)
        ):
            raise ValueError(
                "proposal source names must be safe non-reserved path components"
            )
        if name in normalized:
            raise ValueError(f"duplicate proposal source name: {name}")
        normalized[name] = Path(raw_path)
    return normalized


def _validated_thresholds(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{name} must not be empty")
    normalized = tuple(validate_unit_interval(value, name=name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _truth_paths(truth_dir: Path) -> list[Path]:
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth directory does not exist: {truth_dir}")
    if not truth_dir.is_dir():
        raise NotADirectoryError(f"truth path is not a directory: {truth_dir}")
    paths = sorted(truth_dir.glob("*.txt"))
    if not paths:
        raise ValueError(f"truth directory contains no .txt files: {truth_dir}")
    return paths


def _reject_output_aliases(
    proposal_paths: Mapping[str, Path],
    truth_dir: Path,
    output_dir: Path,
) -> None:
    output = output_dir.resolve()
    if output == truth_dir.resolve():
        raise ValueError("output directory must differ from truth directory")
    for name, path in proposal_paths.items():
        if path.is_dir() and output == path.resolve():
            raise ValueError(
                f"output directory must differ from proposal directory '{name}'"
            )


def _load_source(
    path: Path,
    selected_sequences: tuple[str, ...],
    all_truth_names: set[str],
) -> dict[str, tuple[Detection, ...]]:
    texts = prediction_texts(path)
    unexpected = sorted(Path(name).stem for name in texts if Path(name).stem not in all_truth_names)
    if unexpected:
        raise ValueError(
            "proposal input contains unknown sequence files: " + ", ".join(unexpected)
        )
    rows: dict[str, tuple[Detection, ...]] = {}
    for sequence in selected_sequences:
        parsed = parse_detection_text(
            texts.get(f"{sequence}.txt", ""),
            source=f"{path}:{sequence}.txt",
        )
        for row in parsed:
            if not 0.0 <= row.confidence <= 1.0:
                raise ValueError(
                    f"{path}:{sequence}.txt: proposal confidence must be in [0, 1]"
                )
        rows[sequence] = tuple(parsed)
    return rows


def _canonicalize_proposals(rows: Iterable[Detection]) -> list[Detection]:
    by_frame: dict[int, list[Detection]] = {}
    for row in rows:
        by_frame.setdefault(row.frame_id, []).append(row)
    output: list[Detection] = []
    for frame_id in sorted(by_frame):
        frame_rows = sorted(
            by_frame[frame_id],
            key=lambda row: (
                -row.confidence,
                row.x1,
                row.y1,
                row.width,
                row.height,
                row.object_id,
            ),
        )
        output.extend(
            replace(row, object_id=proposal_id)
            for proposal_id, row in enumerate(frame_rows, start=1)
        )
    return output


def _coverage_counts(
    truth_rows: list[Detection],
    proposal_rows: tuple[Detection, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> _CoverageCounts:
    truth_frames = rows_by_frame(truth_rows)
    proposal_frames = rows_by_frame(
        [row for row in proposal_rows if row.confidence >= confidence_threshold]
    )
    truth_count = len(truth_rows)
    proposal_count = sum(len(rows) for rows in proposal_frames.values())
    matched_count = 0
    matched_iou_sum = 0.0
    best_iou_sum = 0.0
    size_truth = {name: 0 for name, _lower, _upper in _SIZE_BINS}
    size_matched = {name: 0 for name, _lower, _upper in _SIZE_BINS}
    frame_ids = sorted(set(truth_frames) | set(proposal_frames))
    for frame_id in frame_ids:
        truth = truth_frames.get(frame_id, ())
        proposals = proposal_frames.get(frame_id, ())
        matrix = _iou_matrix(truth, proposals)
        if len(truth):
            if matrix.shape[1]:
                best_iou_sum += float(np.max(matrix, axis=1).sum())
            for row in truth:
                size_truth[_size_bin(row)] += 1
        matches = _optimal_matches(matrix, min_iou=iou_threshold)
        matched_count += len(matches)
        matched_iou_sum += sum(match.iou for match in matches)
        for match in matches:
            size_matched[_size_bin(truth[match.truth_index])] += 1
    return _CoverageCounts(
        truth_count=truth_count,
        proposal_count=proposal_count,
        matched_count=matched_count,
        matched_iou_sum=matched_iou_sum,
        best_iou_sum=best_iou_sum,
        size_truth=size_truth,
        size_matched=size_matched,
    )


def _iou_matrix(
    truth: Sequence[Detection],
    proposals: Sequence[Detection],
) -> np.ndarray:
    matrix = np.zeros((len(truth), len(proposals)), dtype=float)
    for truth_index, truth_row in enumerate(truth):
        for proposal_index, proposal_row in enumerate(proposals):
            matrix[truth_index, proposal_index] = box_iou(truth_row, proposal_row)
    return matrix


def _optimal_matches(matrix: np.ndarray, *, min_iou: float) -> tuple[_Match, ...]:
    if matrix.size == 0:
        return ()
    valid = (matrix >= min_iou) & (matrix > _EPS)
    if not valid.any():
        return ()
    cardinality_bonus = float(min(matrix.shape) + 1)
    score = np.where(valid, cardinality_bonus + matrix, 0.0)
    truth_indices, proposal_indices = linear_sum_assignment(-score)
    return tuple(
        _Match(
            truth_index=int(truth_index),
            proposal_index=int(proposal_index),
            iou=float(matrix[truth_index, proposal_index]),
        )
        for truth_index, proposal_index in zip(
            truth_indices,
            proposal_indices,
            strict=True,
        )
        if valid[truth_index, proposal_index]
    )


def _combine_counts(rows: Sequence[_CoverageCounts]) -> _CoverageCounts:
    size_truth = {name: 0 for name, _lower, _upper in _SIZE_BINS}
    size_matched = {name: 0 for name, _lower, _upper in _SIZE_BINS}
    for row in rows:
        for name in size_truth:
            size_truth[name] += row.size_truth[name]
            size_matched[name] += row.size_matched[name]
    return _CoverageCounts(
        truth_count=sum(row.truth_count for row in rows),
        proposal_count=sum(row.proposal_count for row in rows),
        matched_count=sum(row.matched_count for row in rows),
        matched_iou_sum=sum(row.matched_iou_sum for row in rows),
        best_iou_sum=sum(row.best_iou_sum for row in rows),
        size_truth=size_truth,
        size_matched=size_matched,
    )


def _coverage_row(
    source: str,
    confidence_threshold: float,
    iou_threshold: float,
    sequence_count: int,
    counts: _CoverageCounts,
) -> CoverageRow:
    return CoverageRow(
        source=source,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        sequence_count=sequence_count,
        truth_count=counts.truth_count,
        proposal_count=counts.proposal_count,
        matched_count=counts.matched_count,
        recall=_ratio(counts.matched_count, counts.truth_count),
        mean_matched_iou=_ratio(counts.matched_iou_sum, counts.matched_count),
        mean_best_iou=_ratio(counts.best_iou_sum, counts.truth_count),
    )


def _sequence_coverage_row(
    source: str,
    sequence: str,
    confidence_threshold: float,
    iou_threshold: float,
    counts: _CoverageCounts,
) -> SequenceCoverageRow:
    return SequenceCoverageRow(
        source=source,
        sequence=sequence,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        truth_count=counts.truth_count,
        proposal_count=counts.proposal_count,
        matched_count=counts.matched_count,
        recall=_ratio(counts.matched_count, counts.truth_count),
        mean_matched_iou=_ratio(counts.matched_iou_sum, counts.matched_count),
        mean_best_iou=_ratio(counts.best_iou_sum, counts.truth_count),
    )


def _size_bin(row: Detection) -> str:
    area = row.width * row.height
    for name, lower, upper in _SIZE_BINS:
        if lower <= area < upper:
            return name
    raise AssertionError("unreachable size bin")


def _materialize_oracle(
    source: str,
    proposals_by_sequence: Mapping[str, tuple[Detection, ...]],
    truth_by_sequence: Mapping[str, list[Detection]],
    truth_dir: Path,
    output_dir: Path,
    selected_sequences: tuple[str, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
) -> OracleScore:
    oracle_dir = output_dir / "oracle_predictions" / source
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in oracle_dir.glob("*.txt"):
        stale_path.unlink()
    output_rows = 0
    for sequence in selected_sequences:
        rows = _oracle_rows(
            truth_by_sequence[sequence],
            proposals_by_sequence[sequence],
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            exact_seed_rows=True,
        )
        output_rows += len(rows)
        (oracle_dir / f"{sequence}.txt").write_text(
            "".join(format_detection(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    metrics = evaluate_lts_predictions(
        oracle_dir,
        truth_dir,
        sequences=selected_sequences,
    )
    return OracleScore(
        source=source,
        confidence_threshold=confidence_threshold,
        assignment_iou_threshold=iou_threshold,
        exact_seed_rows=True,
        output_dir=str(oracle_dir),
        output_rows=output_rows,
        codabench_hota=metrics.codabench_hota,
        codabench_mota=metrics.codabench_mota,
        codabench_idf1=metrics.codabench_idf1,
        hota=metrics.hota,
        deta=metrics.deta,
        assa=metrics.assa,
        loca=metrics.loca,
    )


def _oracle_rows(
    truth_rows: list[Detection],
    proposal_rows: tuple[Detection, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    exact_seed_rows: bool,
) -> tuple[Detection, ...]:
    truth_frames = rows_by_frame(truth_rows)
    proposal_frames = rows_by_frame(
        [row for row in proposal_rows if row.confidence >= confidence_threshold]
    )
    output: list[Detection] = []
    for frame_id in sorted(truth_frames):
        truth = truth_frames[frame_id]
        if exact_seed_rows and frame_id == 1:
            output.extend(truth)
            continue
        proposals = proposal_frames.get(frame_id, ())
        matches = _optimal_matches(
            _iou_matrix(truth, proposals),
            min_iou=iou_threshold,
        )
        for match in matches:
            truth_row = truth[match.truth_index]
            proposal_row = proposals[match.proposal_index]
            output.append(
                replace(
                    proposal_row,
                    object_id=truth_row.object_id,
                    class_id=truth_row.class_id,
                    visibility=truth_row.visibility,
                )
            )
    return tuple(sorted(output, key=lambda row: (row.frame_id, row.object_id)))


def _write_artifacts(summary: ProposalOracleSummary, output_dir: Path) -> None:
    (output_dir / "proposal_oracle_summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "coverage.csv", summary.coverage)
    _write_csv(output_dir / "sequence_coverage.csv", summary.sequence_coverage)
    _write_csv(output_dir / "size_coverage.csv", summary.size_coverage)
    _write_csv(output_dir / "oracle_scores.csv", summary.oracle_scores)


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    payload = [asdict(row) for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _parse_proposal_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("proposal specifications must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name or not raw_path:
            raise ValueError("proposal specifications must use non-empty NAME=PATH")
        if name in result:
            raise ValueError(f"duplicate proposal source name: {name}")
        result[name] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="proposal directory or ZIP; repeat to build a fused union",
    )
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--confidence-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLDS,
    )
    parser.add_argument(
        "--iou-thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_IOU_THRESHOLDS,
    )
    parser.add_argument("--oracle-confidence-threshold", type=float, default=0.003)
    parser.add_argument("--oracle-iou-threshold", type=float, default=0.05)
    parser.add_argument("--no-fused", action="store_true")
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args(argv)
    summary = audit_proposal_banks(
        _parse_proposal_specs(args.proposal),
        args.truth_dir,
        args.output_dir,
        confidence_thresholds=args.confidence_thresholds,
        iou_thresholds=args.iou_thresholds,
        oracle_confidence_threshold=args.oracle_confidence_threshold,
        oracle_iou_threshold=args.oracle_iou_threshold,
        include_fused=not args.no_fused,
        sequences=args.sequences,
    )
    print(f"proposal_oracle_summary={args.output_dir / 'proposal_oracle_summary.json'}")
    for score in summary.oracle_scores:
        print(
            f"proposal_oracle_{score.source}_codabench_hota="
            f"{score.codabench_hota:.12g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
