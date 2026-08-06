"""Audit low-threshold detector proposal banks for Multi-UAV LTS."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._records import (
    Detection,
    format_detection,
    parse_detection_text,
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


class _ProposalReader:
    """Read one proposal sequence at a time from a directory or ZIP."""

    def __init__(self, path: Path, *, truth_names: set[str]) -> None:
        self.path = path
        self._directory_files: dict[str, Path] = {}
        self._zip: ZipFile | None = None
        self._zip_members: dict[str, str] = {}
        if not path.exists():
            raise FileNotFoundError(f"proposal input does not exist: {path}")
        if path.is_dir():
            self._directory_files = {
                candidate.stem: candidate
                for candidate in sorted(path.glob("*.txt"))
                if candidate.is_file()
            }
            names = set(self._directory_files)
        elif path.is_file():
            try:
                self._zip = ZipFile(path)
            except BadZipFile as exc:
                raise ValueError(f"proposal file is not a ZIP archive: {path}") from exc
            for info in self._zip.infolist():
                member = PurePosixPath(info.filename.replace("\\", "/"))
                if info.is_dir() or member.suffix.lower() != ".txt":
                    continue
                if len(member.parts) != 1:
                    raise ValueError(
                        f"proposal ZIP entries must be root-level .txt files: {info.filename}"
                    )
                sequence = member.stem
                if sequence in self._zip_members:
                    raise ValueError(
                        f"proposal ZIP contains duplicate sequence entry: {member.name}"
                    )
                self._zip_members[sequence] = info.filename
            names = set(self._zip_members)
        else:
            raise ValueError(f"unsupported proposal input: {path}")
        unexpected = sorted(names - truth_names)
        if unexpected:
            raise ValueError(
                "proposal input contains unknown sequence files: "
                + ", ".join(unexpected)
            )

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def rows(self, sequence: str) -> tuple[Detection, ...]:
        if self._zip is None:
            candidate = self._directory_files.get(sequence)
            text = "" if candidate is None else candidate.read_text(encoding="utf-8")
        else:
            member = self._zip_members.get(sequence)
            text = "" if member is None else self._zip.read(member).decode("utf-8")
        source = f"{self.path}:{sequence}.txt"
        try:
            parsed = parse_detection_text(text, source=source)
        except ValueError as exc:
            if "confidence must be in [-1, 1]" not in str(exc):
                raise
            raise ValueError(
                f"{source}: proposal confidence must be in [0, 1]"
            ) from exc
        for row in parsed:
            if not 0.0 <= row.confidence <= 1.0:
                raise ValueError(
                    f"{source}: proposal confidence must be in [0, 1]"
                )
        return tuple(parsed)


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

    source_names = list(normalized_paths)
    if include_fused and len(source_names) > 1:
        source_names.append("fused")
    sources = [
        ProposalSource(name=name, path=str(path), fused=False)
        for name, path in normalized_paths.items()
    ]
    if "fused" in source_names:
        sources.append(ProposalSource(name="fused", path="<union>", fused=True))

    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_dirs = _prepare_oracle_dirs(output_dir, source_names)
    output_rows = {source: 0 for source in source_names}
    count_groups: dict[tuple[str, float, float], list[_CoverageCounts]] = {
        (source, confidence, iou): []
        for source in source_names
        for confidence in confidence_values
        for iou in iou_values
    }
    sequence_rows: list[SequenceCoverageRow] = []

    with ExitStack() as stack:
        readers: dict[str, _ProposalReader] = {}
        for name, path in normalized_paths.items():
            reader = _ProposalReader(path, truth_names=truth_names)
            stack.callback(reader.close)
            readers[name] = reader
        for truth_path in selected_paths:
            sequence = truth_path.stem
            truth_rows = parse_detection_text(
                truth_path.read_text(encoding="utf-8"),
                source=str(truth_path),
            )
            rows_by_source = {
                name: tuple(_canonicalize_proposals(reader.rows(sequence)))
                for name, reader in readers.items()
            }
            if "fused" in source_names:
                rows_by_source["fused"] = tuple(
                    _canonicalize_proposals(
                        row
                        for name in normalized_paths
                        for row in rows_by_source[name]
                    )
                )
            for source in source_names:
                proposal_rows = rows_by_source[source]
                grid = _coverage_grid(
                    truth_rows,
                    proposal_rows,
                    confidence_thresholds=confidence_values,
                    iou_thresholds=iou_values,
                )
                for confidence in confidence_values:
                    for iou in iou_values:
                        counts = grid[(confidence, iou)]
                        count_groups[(source, confidence, iou)].append(counts)
                        sequence_rows.append(
                            _sequence_coverage_row(
                                source,
                                sequence,
                                confidence,
                                iou,
                                counts,
                            )
                        )
                oracle_rows = _oracle_rows(
                    truth_rows,
                    proposal_rows,
                    confidence_threshold=oracle_confidence,
                    iou_threshold=oracle_iou,
                    exact_seed_rows=True,
                )
                output_rows[source] += len(oracle_rows)
                (oracle_dirs[source] / f"{sequence}.txt").write_text(
                    "".join(format_detection(row) + "\n" for row in oracle_rows),
                    encoding="utf-8",
                )

    coverage_rows: list[CoverageRow] = []
    size_rows: list[SizeCoverageRow] = []
    for source in source_names:
        for confidence in confidence_values:
            for iou in iou_values:
                aggregate = _combine_counts(count_groups[(source, confidence, iou)])
                coverage_rows.append(
                    _coverage_row(
                        source,
                        confidence,
                        iou,
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
                            confidence_threshold=confidence,
                            iou_threshold=iou,
                            size_bin=size_bin,
                            truth_count=truth_count,
                            matched_count=matched_count,
                            recall=_ratio(matched_count, truth_count),
                        )
                    )

    oracle_scores = tuple(
        _score_oracle(
            source,
            oracle_dirs[source],
            truth_dir,
            selected_sequences,
            confidence_threshold=oracle_confidence,
            iou_threshold=oracle_iou,
            output_rows=output_rows[source],
        )
        for source in source_names
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
    truth = truth_dir.resolve()
    if output == truth:
        raise ValueError("output directory must differ from truth directory")
    seen_inputs: dict[Path, str] = {}
    for name, path in proposal_paths.items():
        resolved = path.resolve()
        if resolved == truth:
            raise ValueError(f"proposal source '{name}' must not alias the truth directory")
        if resolved in seen_inputs:
            raise ValueError(
                f"proposal sources '{seen_inputs[resolved]}' and '{name}' alias the same input"
            )
        seen_inputs[resolved] = name
        if path.is_dir() and output == resolved:
            raise ValueError(
                f"output directory must differ from proposal directory '{name}'"
            )


def _prepare_oracle_dirs(
    output_dir: Path,
    sources: Sequence[str],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source in sources:
        oracle_dir = output_dir / "oracle_predictions" / source
        oracle_dir.mkdir(parents=True, exist_ok=True)
        for stale_path in oracle_dir.glob("*.txt"):
            stale_path.unlink()
        result[source] = oracle_dir
    return result


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


def _coverage_grid(
    truth_rows: list[Detection],
    proposal_rows: tuple[Detection, ...],
    *,
    confidence_thresholds: tuple[float, ...],
    iou_thresholds: tuple[float, ...],
) -> dict[tuple[float, float], _CoverageCounts]:
    truth_frames = rows_by_frame(truth_rows)
    proposal_frames = rows_by_frame(proposal_rows)
    size_truth = Counter(_size_bin(row) for row in truth_rows)
    proposal_counts = {confidence: 0 for confidence in confidence_thresholds}
    best_iou_sums = {confidence: 0.0 for confidence in confidence_thresholds}
    matched_counts = {
        (confidence, iou): 0
        for confidence in confidence_thresholds
        for iou in iou_thresholds
    }
    matched_iou_sums = {key: 0.0 for key in matched_counts}
    size_matched = {key: Counter() for key in matched_counts}
    frame_ids = sorted(set(truth_frames) | set(proposal_frames))
    for frame_id in frame_ids:
        truth = truth_frames.get(frame_id, ())
        proposals = proposal_frames.get(frame_id, ())
        matrix = _iou_matrix(truth, proposals)
        scores = np.asarray([row.confidence for row in proposals], dtype=float)
        for confidence in confidence_thresholds:
            indices = np.flatnonzero(scores >= confidence)
            proposal_counts[confidence] += int(indices.size)
            filtered = matrix[:, indices]
            if len(truth) and filtered.shape[1]:
                best_iou_sums[confidence] += float(np.max(filtered, axis=1).sum())
            for iou in iou_thresholds:
                key = (confidence, iou)
                matches = _optimal_matches(filtered, min_iou=iou)
                matched_counts[key] += len(matches)
                matched_iou_sums[key] += sum(match.iou for match in matches)
                for match in matches:
                    size_matched[key][_size_bin(truth[match.truth_index])] += 1
    result: dict[tuple[float, float], _CoverageCounts] = {}
    for confidence in confidence_thresholds:
        for iou in iou_thresholds:
            key = (confidence, iou)
            result[key] = _CoverageCounts(
                truth_count=len(truth_rows),
                proposal_count=proposal_counts[confidence],
                matched_count=matched_counts[key],
                matched_iou_sum=matched_iou_sums[key],
                best_iou_sum=best_iou_sums[confidence],
                size_truth={
                    name: int(size_truth[name]) for name, _lower, _upper in _SIZE_BINS
                },
                size_matched={
                    name: int(size_matched[key][name])
                    for name, _lower, _upper in _SIZE_BINS
                },
            )
    return result


def _iou_matrix(
    truth: Sequence[Detection],
    proposals: Sequence[Detection],
) -> np.ndarray:
    if not truth or not proposals:
        return np.zeros((len(truth), len(proposals)), dtype=float)
    truth_boxes = np.asarray(
        [
            [row.x1, row.y1, row.x1 + row.width, row.y1 + row.height]
            for row in truth
        ],
        dtype=float,
    )
    proposal_boxes = np.asarray(
        [
            [row.x1, row.y1, row.x1 + row.width, row.y1 + row.height]
            for row in proposals
        ],
        dtype=float,
    )
    top_left = np.maximum(truth_boxes[:, None, :2], proposal_boxes[None, :, :2])
    bottom_right = np.minimum(truth_boxes[:, None, 2:], proposal_boxes[None, :, 2:])
    intersection_wh = np.maximum(0.0, bottom_right - top_left)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    truth_area = (
        (truth_boxes[:, 2] - truth_boxes[:, 0])
        * (truth_boxes[:, 3] - truth_boxes[:, 1])
    )[:, None]
    proposal_area = (
        (proposal_boxes[:, 2] - proposal_boxes[:, 0])
        * (proposal_boxes[:, 3] - proposal_boxes[:, 1])
    )[None, :]
    union = truth_area + proposal_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


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


def _score_oracle(
    source: str,
    oracle_dir: Path,
    truth_dir: Path,
    selected_sequences: tuple[str, ...],
    *,
    confidence_threshold: float,
    iou_threshold: float,
    output_rows: int,
) -> OracleScore:
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
