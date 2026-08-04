"""Guarded candidate selection for Multi-UAV LTS train-split experiments.

The raw prediction set is always an explicit candidate. A transformed candidate
is selected only when it clears paired uncertainty, secondary-metric, scenario,
coverage, and sequence-manifest guards; otherwise selection falls back to raw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ._records import prediction_texts
from .fixed_population_cv import build_stratified_folds, scenario_prefix
from .metrics import BenchmarkMetrics, SequenceMetrics, evaluate_lts_predictions

_SCHEMA = "raft-uav-multi-uav-lts-guarded-tournament-v1"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class GroupScore:
    """Organizer-compatible score for one fold or scenario group."""

    group: str
    sequences: tuple[str, ...]
    codabench_hota: float
    codabench_mota: float
    codabench_idf1: float
    hota: float
    deta: float
    assa: float
    loca: float
    mota: float
    idf1: float
    id_switches: int
    predicted_detections: int


@dataclass(frozen=True)
class SequenceScore:
    """Per-sequence paired-comparison values."""

    sequence: str
    codabench_hota: float
    codabench_mota: float
    codabench_idf1: float
    hota: float
    deta: float
    assa: float
    loca: float
    id_switches: int
    predicted_detections: int


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's guarded tournament score."""

    rank: int
    name: str
    prediction_path: str
    is_raw: bool
    selected: bool
    eligible: bool
    rejection_reasons: tuple[str, ...]
    sequence_count: int
    prediction_file_count: int
    missing_sequences: tuple[str, ...]
    content_sha256: str
    content_bytes: int
    mean_cv_codabench_hota: float
    std_cv_codabench_hota: float
    mean_cv_codabench_mota: float
    mean_cv_codabench_idf1: float
    pooled_codabench_hota: float
    pooled_codabench_mota: float
    pooled_codabench_idf1: float
    pooled_hota: float
    pooled_deta: float
    pooled_assa: float
    pooled_loca: float
    pooled_mota: float
    pooled_idf1: float
    id_switches: int
    mean_cv_hota_gain_vs_raw: float
    mean_cv_mota_delta_vs_raw: float
    mean_cv_idf1_delta_vs_raw: float
    paired_hota_gain_ci_low: float
    paired_hota_gain_ci_high: float
    worst_scenario_hota_delta_vs_raw: float
    folds: tuple[GroupScore, ...]
    scenarios: tuple[GroupScore, ...]
    sequences: tuple[SequenceScore, ...]


@dataclass(frozen=True)
class TournamentResult:
    """Guarded tournament result and output location."""

    selected_candidate: str
    raw_candidate: str
    sequence_count: int
    folds: tuple[tuple[str, ...], ...]
    rows: tuple[CandidateScore, ...]
    output_dir: str


@dataclass(frozen=True)
class _EvaluatedCandidate:
    name: str
    path: Path
    is_raw: bool
    missing_sequences: tuple[str, ...]
    prediction_file_count: int
    content_sha256: str
    content_bytes: int
    metrics: BenchmarkMetrics
    folds: tuple[GroupScore, ...]
    scenarios: tuple[GroupScore, ...]
    sequences: tuple[SequenceScore, ...]


def run_guarded_tournament(
    raw_prediction_path: Path,
    truth_dir: Path,
    output_dir: Path,
    *,
    candidates: Sequence[tuple[str, Path]] = (),
    fold_count: int = 5,
    seed: int = 0,
    expected_sequence_count: int = 102,
    sequences: Sequence[str] = (),
    bootstrap_samples: int = 5000,
    min_mean_hota_gain: float = 0.001,
    min_ci_hota_gain: float = 0.0,
    max_mean_mota_drop: float = 0.002,
    max_mean_idf1_drop: float = 0.002,
    max_worst_scenario_hota_drop: float = 0.01,
    require_complete: bool = True,
    require_improvement: bool = False,
    copy_selected: bool = True,
) -> TournamentResult:
    """Evaluate, guard, and deterministically select an LTS prediction candidate."""

    fold_count = _integer_at_least(fold_count, "fold_count", 2)
    bootstrap_samples = _integer_at_least(
        bootstrap_samples, "bootstrap_samples", 0
    )
    expected_sequence_count = _integer_at_least(
        expected_sequence_count, "expected_sequence_count", 0
    )
    seed = _integer(seed, "seed")
    min_mean_hota_gain = _nonnegative_float(
        min_mean_hota_gain, "min_mean_hota_gain"
    )
    min_ci_hota_gain = _finite_float(min_ci_hota_gain, "min_ci_hota_gain")
    max_mean_mota_drop = _nonnegative_float(
        max_mean_mota_drop, "max_mean_mota_drop"
    )
    max_mean_idf1_drop = _nonnegative_float(
        max_mean_idf1_drop, "max_mean_idf1_drop"
    )
    max_worst_scenario_hota_drop = _nonnegative_float(
        max_worst_scenario_hota_drop,
        "max_worst_scenario_hota_drop",
    )

    truth_sequences = _truth_sequences(truth_dir)
    selected_sequences = _selected_sequences(truth_sequences, sequences)
    if expected_sequence_count and len(selected_sequences) != expected_sequence_count:
        raise ValueError(
            "selected truth sequence count does not match "
            f"expected_sequence_count={expected_sequence_count}: "
            f"{len(selected_sequences)}"
        )
    if fold_count > len(selected_sequences):
        raise ValueError("fold_count cannot exceed the selected sequence count")

    specs = _candidate_specs(raw_prediction_path, candidates)
    folds = build_stratified_folds(
        selected_sequences,
        fold_count=fold_count,
        seed=seed,
    )
    scenario_groups = _scenario_groups(selected_sequences)
    evaluated = tuple(
        _evaluate_candidate(
            name=name,
            path=path,
            is_raw=is_raw,
            truth_dir=truth_dir,
            sequences=selected_sequences,
            folds=folds,
            scenarios=scenario_groups,
            require_complete=require_complete,
        )
        for name, path, is_raw in specs
    )

    raw = evaluated[0]
    raw_folds = {row.group: row for row in raw.folds}
    raw_scenarios = {row.group: row for row in raw.scenarios}
    raw_sequences = {row.sequence: row for row in raw.sequences}
    scored: list[CandidateScore] = []
    for candidate in evaluated:
        mean_hota = _mean(row.codabench_hota for row in candidate.folds)
        mean_mota = _mean(row.codabench_mota for row in candidate.folds)
        mean_idf1 = _mean(row.codabench_idf1 for row in candidate.folds)
        hota_gain = mean_hota - _mean(
            raw_folds[row.group].codabench_hota for row in candidate.folds
        )
        mota_delta = mean_mota - _mean(
            raw_folds[row.group].codabench_mota for row in candidate.folds
        )
        idf1_delta = mean_idf1 - _mean(
            raw_folds[row.group].codabench_idf1 for row in candidate.folds
        )
        candidate_sequences = {row.sequence: row for row in candidate.sequences}
        ci_low, ci_high = _paired_bootstrap_interval(
            candidate_sequences,
            raw_sequences,
            selected_sequences,
            samples=bootstrap_samples,
            seed=_candidate_seed(seed, candidate.name),
        )
        scenario_deltas = tuple(
            row.codabench_hota - raw_scenarios[row.group].codabench_hota
            for row in candidate.scenarios
        )
        worst_scenario_delta = min(scenario_deltas) if scenario_deltas else 0.0

        reasons: list[str] = []
        if not candidate.is_raw:
            if candidate.missing_sequences:
                reasons.append("incomplete prediction coverage")
            if hota_gain + 1e-15 < min_mean_hota_gain:
                reasons.append(
                    "mean CV CODABENCH_HOTA gain "
                    f"{hota_gain:.6f} < {min_mean_hota_gain:.6f}"
                )
            if ci_low + 1e-15 < min_ci_hota_gain:
                reasons.append(
                    "paired HOTA gain CI lower bound "
                    f"{ci_low:.6f} < {min_ci_hota_gain:.6f}"
                )
            if mota_delta < -max_mean_mota_drop - 1e-15:
                reasons.append(
                    "mean CV CODABENCH_MOTA delta "
                    f"{mota_delta:.6f} < {-max_mean_mota_drop:.6f}"
                )
            if idf1_delta < -max_mean_idf1_drop - 1e-15:
                reasons.append(
                    "mean CV CODABENCH_IDF1 delta "
                    f"{idf1_delta:.6f} < {-max_mean_idf1_drop:.6f}"
                )
            if worst_scenario_delta < -max_worst_scenario_hota_drop - 1e-15:
                reasons.append(
                    "worst-scenario CODABENCH_HOTA delta "
                    f"{worst_scenario_delta:.6f} < "
                    f"{-max_worst_scenario_hota_drop:.6f}"
                )

        scored.append(
            CandidateScore(
                rank=0,
                name=candidate.name,
                prediction_path=str(candidate.path),
                is_raw=candidate.is_raw,
                selected=False,
                eligible=candidate.is_raw or not reasons,
                rejection_reasons=tuple(reasons),
                sequence_count=candidate.metrics.sequence_count,
                prediction_file_count=candidate.prediction_file_count,
                missing_sequences=candidate.missing_sequences,
                content_sha256=candidate.content_sha256,
                content_bytes=candidate.content_bytes,
                mean_cv_codabench_hota=mean_hota,
                std_cv_codabench_hota=_std(
                    row.codabench_hota for row in candidate.folds
                ),
                mean_cv_codabench_mota=mean_mota,
                mean_cv_codabench_idf1=mean_idf1,
                pooled_codabench_hota=candidate.metrics.codabench_hota,
                pooled_codabench_mota=candidate.metrics.codabench_mota,
                pooled_codabench_idf1=candidate.metrics.codabench_idf1,
                pooled_hota=candidate.metrics.hota,
                pooled_deta=candidate.metrics.deta,
                pooled_assa=candidate.metrics.assa,
                pooled_loca=candidate.metrics.loca,
                pooled_mota=candidate.metrics.mota,
                pooled_idf1=candidate.metrics.idf1,
                id_switches=candidate.metrics.id_switches,
                mean_cv_hota_gain_vs_raw=hota_gain,
                mean_cv_mota_delta_vs_raw=mota_delta,
                mean_cv_idf1_delta_vs_raw=idf1_delta,
                paired_hota_gain_ci_low=ci_low,
                paired_hota_gain_ci_high=ci_high,
                worst_scenario_hota_delta_vs_raw=worst_scenario_delta,
                folds=candidate.folds,
                scenarios=candidate.scenarios,
                sequences=candidate.sequences,
            )
        )

    ranked = sorted(scored, key=_ranking_key)
    selected = next(row for row in ranked if row.eligible)
    ranked_rows = tuple(
        replace(row, rank=rank, selected=row.name == selected.name)
        for rank, row in enumerate(ranked, start=1)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_selected:
        _copy_selected_predictions(Path(selected.prediction_path), output_dir)
    result = TournamentResult(
        selected_candidate=selected.name,
        raw_candidate="raw",
        sequence_count=len(selected_sequences),
        folds=folds,
        rows=ranked_rows,
        output_dir=str(output_dir),
    )
    _write_outputs(
        result,
        truth_dir=truth_dir,
        selected_sequences=selected_sequences,
        scenario_groups=scenario_groups,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        thresholds={
            "min_mean_hota_gain": min_mean_hota_gain,
            "min_ci_hota_gain": min_ci_hota_gain,
            "max_mean_mota_drop": max_mean_mota_drop,
            "max_mean_idf1_drop": max_mean_idf1_drop,
            "max_worst_scenario_hota_drop": max_worst_scenario_hota_drop,
            "require_complete": require_complete,
            "require_improvement": require_improvement,
        },
    )
    if require_improvement and selected.is_raw:
        raise RuntimeError(
            "no transformed candidate cleared the configured guards; "
            "raw fallback selected"
        )
    return result


def _candidate_specs(
    raw_prediction_path: Path,
    candidates: Sequence[tuple[str, Path]],
) -> tuple[tuple[str, Path, bool], ...]:
    specs: list[tuple[str, Path, bool]] = [
        ("raw", Path(raw_prediction_path).expanduser(), True)
    ]
    seen = {"raw"}
    for raw_name, raw_path in candidates:
        name = str(raw_name).strip()
        if not _NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "candidate names must match [A-Za-z0-9][A-Za-z0-9_.-]*"
            )
        if name in seen:
            raise ValueError(f"duplicate candidate name: {name}")
        seen.add(name)
        specs.append((name, Path(raw_path).expanduser(), False))
    for name, path, _ in specs:
        if not path.exists():
            raise FileNotFoundError(f"candidate {name!r} does not exist: {path}")
        if not (path.is_dir() or path.is_file()):
            raise ValueError(f"candidate {name!r} is not a file or directory: {path}")
    return tuple(specs)


def _evaluate_candidate(
    *,
    name: str,
    path: Path,
    is_raw: bool,
    truth_dir: Path,
    sequences: tuple[str, ...],
    folds: tuple[tuple[str, ...], ...],
    scenarios: tuple[tuple[str, tuple[str, ...]], ...],
    require_complete: bool,
) -> _EvaluatedCandidate:
    texts = prediction_texts(path)
    prediction_names = {Path(filename).stem for filename in texts}
    missing = tuple(sorted(set(sequences) - prediction_names))
    if missing and (is_raw or require_complete):
        raise ValueError(
            f"candidate {name!r} is missing {len(missing)} selected sequence files: "
            + ", ".join(missing[:10])
        )
    metrics = evaluate_lts_predictions(path, truth_dir, sequences=sequences)
    fold_scores = tuple(
        _evaluate_group(
            path,
            truth_dir,
            group=f"fold_{index}",
            sequences=fold,
        )
        for index, fold in enumerate(folds)
    )
    scenario_scores = tuple(
        _evaluate_group(
            path,
            truth_dir,
            group=prefix,
            sequences=group_sequences,
        )
        for prefix, group_sequences in scenarios
    )
    digest, content_bytes = _content_digest(path)
    return _EvaluatedCandidate(
        name=name,
        path=path,
        is_raw=is_raw,
        missing_sequences=missing,
        prediction_file_count=len(texts),
        content_sha256=digest,
        content_bytes=content_bytes,
        metrics=metrics,
        folds=fold_scores,
        scenarios=scenario_scores,
        sequences=tuple(_sequence_score(row) for row in metrics.sequences),
    )


def _evaluate_group(
    prediction_path: Path,
    truth_dir: Path,
    *,
    group: str,
    sequences: tuple[str, ...],
) -> GroupScore:
    metrics = evaluate_lts_predictions(
        prediction_path,
        truth_dir,
        sequences=sequences,
    )
    return GroupScore(
        group=group,
        sequences=sequences,
        codabench_hota=metrics.codabench_hota,
        codabench_mota=metrics.codabench_mota,
        codabench_idf1=metrics.codabench_idf1,
        hota=metrics.hota,
        deta=metrics.deta,
        assa=metrics.assa,
        loca=metrics.loca,
        mota=metrics.mota,
        idf1=metrics.idf1,
        id_switches=metrics.id_switches,
        predicted_detections=metrics.predicted_detections,
    )


def _sequence_score(row: SequenceMetrics) -> SequenceScore:
    return SequenceScore(
        sequence=row.sequence,
        codabench_hota=row.hota_at_005,
        codabench_mota=row.mota,
        codabench_idf1=row.idf1,
        hota=row.hota,
        deta=row.deta,
        assa=row.assa,
        loca=row.loca,
        id_switches=row.id_switches,
        predicted_detections=row.predicted_detections,
    )


def _paired_bootstrap_interval(
    candidate: dict[str, SequenceScore],
    raw: dict[str, SequenceScore],
    sequences: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    differences = np.asarray(
        [
            candidate[name].codabench_hota - raw[name].codabench_hota
            for name in sequences
        ],
        dtype=float,
    )
    if differences.size == 0:
        raise ValueError("paired bootstrap requires at least one sequence")
    mean = float(np.mean(differences))
    if samples == 0 or differences.size == 1:
        return mean, mean
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        differences.size,
        size=(samples, differences.size),
    )
    bootstrap_means = np.mean(differences[indices], axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def _ranking_key(row: CandidateScore) -> tuple[object, ...]:
    return (
        not row.eligible,
        -row.mean_cv_codabench_hota,
        -row.paired_hota_gain_ci_low,
        -row.mean_cv_codabench_idf1,
        -row.mean_cv_codabench_mota,
        0 if row.is_raw else 1,
        row.name,
    )


def _truth_sequences(truth_dir: Path) -> tuple[str, ...]:
    if not truth_dir.exists():
        raise FileNotFoundError(f"truth directory does not exist: {truth_dir}")
    if not truth_dir.is_dir():
        raise NotADirectoryError(f"truth path is not a directory: {truth_dir}")
    sequences = tuple(path.stem for path in sorted(truth_dir.glob("*.txt")))
    if not sequences:
        raise ValueError(f"truth directory contains no .txt files: {truth_dir}")
    return sequences


def _selected_sequences(
    available: tuple[str, ...],
    requested: Sequence[str],
) -> tuple[str, ...]:
    if not requested:
        return available
    selected = tuple(str(value).strip() for value in requested)
    if any(not value for value in selected):
        raise ValueError("sequence names must be non-empty")
    if len(set(selected)) != len(selected):
        raise ValueError("sequence names must be unique")
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"unknown truth sequences: {', '.join(missing)}")
    return tuple(sorted(selected))


def _scenario_groups(
    sequences: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups: dict[str, list[str]] = {}
    for sequence in sequences:
        groups.setdefault(scenario_prefix(sequence), []).append(sequence)
    return tuple(
        (prefix, tuple(sorted(values)))
        for prefix, values in sorted(groups.items())
    )


def _content_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    if path.is_file():
        files = (path,)
        root = path.parent
    else:
        files = tuple(child for child in sorted(path.rglob("*")) if child.is_file())
        root = path
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                total_bytes += len(chunk)
                digest.update(chunk)
    return digest.hexdigest(), total_bytes


def _copy_selected_predictions(path: Path, output_dir: Path) -> None:
    output_resolved = output_dir.resolve()
    path_resolved = path.resolve()
    if output_resolved == path_resolved or output_resolved.is_relative_to(path_resolved):
        raise ValueError("output_dir must not be inside the selected candidate path")
    if path.is_dir():
        target = output_dir / "selected_predictions"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
        return
    suffix = "".join(path.suffixes) or ".bin"
    target = output_dir / f"selected_predictions{suffix}"
    if target.exists():
        target.unlink()
    shutil.copy2(path, target)


def _write_outputs(
    result: TournamentResult,
    *,
    truth_dir: Path,
    selected_sequences: tuple[str, ...],
    scenario_groups: tuple[tuple[str, tuple[str, ...]], ...],
    seed: int,
    bootstrap_samples: int,
    thresholds: dict[str, object],
) -> None:
    output_dir = Path(result.output_dir)
    payload = {
        "schema": _SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "truth_dir": str(truth_dir),
        "selected_candidate": result.selected_candidate,
        "selection_status": (
            "raw_fallback"
            if result.selected_candidate == result.raw_candidate
            else "transformed_candidate_selected"
        ),
        "raw_candidate": result.raw_candidate,
        "sequence_count": result.sequence_count,
        "sequences": list(selected_sequences),
        "folds": [list(fold) for fold in result.folds],
        "scenario_groups": {
            prefix: list(values) for prefix, values in scenario_groups
        },
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "thresholds": thresholds,
        "rows": [asdict(row) for row in result.rows],
    }
    (output_dir / "tournament_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "selected_candidate.txt").write_text(
        result.selected_candidate + "\n",
        encoding="utf-8",
    )
    _write_ranking_csv(output_dir, result.rows)
    _write_group_csv(output_dir, result.rows)
    _write_sequence_delta_csv(output_dir, result.rows, selected_sequences)
    _write_provenance(output_dir, result.rows)


def _write_ranking_csv(
    output_dir: Path,
    rows: tuple[CandidateScore, ...],
) -> None:
    excluded = {"folds", "scenarios", "sequences"}
    fields = [
        field
        for field in CandidateScore.__dataclass_fields__
        if field not in excluded
    ]
    with (output_dir / "tournament_ranking.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            values["rejection_reasons"] = " | ".join(row.rejection_reasons)
            values["missing_sequences"] = " ".join(row.missing_sequences)
            writer.writerow({field: values[field] for field in fields})


def _write_group_csv(
    output_dir: Path,
    rows: tuple[CandidateScore, ...],
) -> None:
    fields = ["candidate", "group_type", *GroupScore.__dataclass_fields__]
    with (output_dir / "group_scores.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for group_type, groups in (
                ("fold", row.folds),
                ("scenario", row.scenarios),
            ):
                for group in groups:
                    values = asdict(group)
                    values["sequences"] = " ".join(group.sequences)
                    writer.writerow(
                        {
                            "candidate": row.name,
                            "group_type": group_type,
                            **values,
                        }
                    )


def _write_sequence_delta_csv(
    output_dir: Path,
    rows: tuple[CandidateScore, ...],
    selected_sequences: tuple[str, ...],
) -> None:
    raw = next(row for row in rows if row.is_raw)
    raw_values = {row.sequence: row for row in raw.sequences}
    fields = [
        "candidate",
        "sequence",
        "codabench_hota",
        "raw_codabench_hota",
        "hota_delta",
        "codabench_mota",
        "raw_codabench_mota",
        "mota_delta",
        "codabench_idf1",
        "raw_codabench_idf1",
        "idf1_delta",
    ]
    with (output_dir / "sequence_deltas.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in rows:
            values = {row.sequence: row for row in candidate.sequences}
            for sequence in selected_sequences:
                current = values[sequence]
                baseline = raw_values[sequence]
                writer.writerow(
                    {
                        "candidate": candidate.name,
                        "sequence": sequence,
                        "codabench_hota": current.codabench_hota,
                        "raw_codabench_hota": baseline.codabench_hota,
                        "hota_delta": (
                            current.codabench_hota - baseline.codabench_hota
                        ),
                        "codabench_mota": current.codabench_mota,
                        "raw_codabench_mota": baseline.codabench_mota,
                        "mota_delta": (
                            current.codabench_mota - baseline.codabench_mota
                        ),
                        "codabench_idf1": current.codabench_idf1,
                        "raw_codabench_idf1": baseline.codabench_idf1,
                        "idf1_delta": (
                            current.codabench_idf1 - baseline.codabench_idf1
                        ),
                    }
                )


def _write_provenance(
    output_dir: Path,
    rows: tuple[CandidateScore, ...],
) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-tournament-provenance-v1",
        "git_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "candidates": [
            {
                "name": row.name,
                "prediction_path": row.prediction_path,
                "content_sha256": row.content_sha256,
                "content_bytes": row.content_bytes,
                "prediction_file_count": row.prediction_file_count,
                "missing_sequences": list(row.missing_sequences),
                "selected": row.selected,
            }
            for row in rows
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return float(statistics.fmean(materialized))


def _std(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if len(materialized) < 2:
        return 0.0
    return float(statistics.pstdev(materialized))


def _candidate_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        array = np.asarray(value)
        if array.ndim != 0:
            raise TypeError
        scalar = array.item()
        if isinstance(scalar, complex):
            if scalar.imag != 0.0:
                raise TypeError
            scalar = scalar.real
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _nonnegative_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _integer(value: object, name: str) -> int:
    number = _finite_float(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _integer_at_least(value: object, name: str, minimum: int) -> int:
    integer = _integer(value, name)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def _parse_candidate_specs(
    values: Sequence[str],
) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"candidate specification must be NAME=PATH, got: {value!r}"
            )
        name, raw_path = value.split("=", 1)
        if not raw_path.strip():
            raise ValueError(f"candidate path is empty for {name!r}")
        parsed.append((name.strip(), Path(raw_path.strip())))
    return tuple(parsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_prediction_path", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-sequence-count", type=int, default=102)
    parser.add_argument("--sequences", nargs="*", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--min-mean-hota-gain", type=float, default=0.001)
    parser.add_argument("--min-ci-hota-gain", type=float, default=0.0)
    parser.add_argument("--max-mean-mota-drop", type=float, default=0.002)
    parser.add_argument("--max-mean-idf1-drop", type=float, default=0.002)
    parser.add_argument(
        "--max-worst-scenario-hota-drop",
        type=float,
        default=0.01,
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-improvement", action="store_true")
    parser.add_argument("--no-copy-selected", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = run_guarded_tournament(
            args.raw_prediction_path,
            args.truth_dir,
            args.output_dir,
            candidates=_parse_candidate_specs(args.candidate),
            fold_count=args.fold_count,
            seed=args.seed,
            expected_sequence_count=args.expected_sequence_count,
            sequences=args.sequences,
            bootstrap_samples=args.bootstrap_samples,
            min_mean_hota_gain=args.min_mean_hota_gain,
            min_ci_hota_gain=args.min_ci_hota_gain,
            max_mean_mota_drop=args.max_mean_mota_drop,
            max_mean_idf1_drop=args.max_mean_idf1_drop,
            max_worst_scenario_hota_drop=args.max_worst_scenario_hota_drop,
            require_complete=not args.allow_incomplete,
            require_improvement=args.require_improvement,
            copy_selected=not args.no_copy_selected,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    selected = next(row for row in result.rows if row.selected)
    print(f"selected_candidate={selected.name}")
    print(f"selection_eligible={int(selected.eligible)}")
    print(f"mean_cv_CODABENCH_HOTA={selected.mean_cv_codabench_hota:.6f}")
    print(f"mean_cv_CODABENCH_MOTA={selected.mean_cv_codabench_mota:.6f}")
    print(f"mean_cv_CODABENCH_IDF1={selected.mean_cv_codabench_idf1:.6f}")
    print(f"output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
