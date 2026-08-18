"""Guarded scenario-level expert selection for Multi-UAV LTS predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

_SCORE_SCHEMA = "raft-uav-multi-uav-lts-scenario-score-bank-v1"
_POLICY_SCHEMA = "raft-uav-multi-uav-lts-scenario-expert-policy-v1"
_SUMMARY_SCHEMA = "raft-uav-multi-uav-lts-scenario-expert-output-v1"
_CANDIDATE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_SEQUENCE_SUFFIX = re.compile(r"^(?P<prefix>.+)_\d+$")


@dataclass(frozen=True)
class CandidateScore:
    sequence: str
    candidate: str
    hota: float
    mota: float
    idf1: float


@dataclass(frozen=True)
class GateConfig:
    raw_candidate: str = "raw"
    fold_count: int = 5
    seed: int = 0
    min_prefix_samples: int = 3
    min_train_hota_gain: float = 0.001
    prior_strength: float = 3.0
    max_train_mota_drop: float = 0.002
    max_train_idf1_drop: float = 0.002
    min_cv_hota_gain: float = 0.0005
    max_cv_mota_drop: float = 0.002
    max_cv_idf1_drop: float = 0.002
    max_worst_prefix_hota_drop: float = 0.005

    def validate(self) -> None:
        if not _CANDIDATE_PATTERN.fullmatch(self.raw_candidate):
            raise ValueError("raw candidate has an invalid name")
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if self.min_prefix_samples <= 0:
            raise ValueError("min_prefix_samples must be positive")
        if self.prior_strength < 0 or not math.isfinite(self.prior_strength):
            raise ValueError("prior_strength must be finite and non-negative")
        non_negative = {
            "min_train_hota_gain": self.min_train_hota_gain,
            "max_train_mota_drop": self.max_train_mota_drop,
            "max_train_idf1_drop": self.max_train_idf1_drop,
            "min_cv_hota_gain": self.min_cv_hota_gain,
            "max_cv_mota_drop": self.max_cv_mota_drop,
            "max_cv_idf1_drop": self.max_cv_idf1_drop,
            "max_worst_prefix_hota_drop": self.max_worst_prefix_hota_drop,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def sequence_prefix(sequence: str) -> str:
    name = Path(sequence).stem.strip()
    if not name:
        raise ValueError("sequence name must not be empty")
    match = _SEQUENCE_SUFFIX.fullmatch(name)
    return match.group("prefix") if match else name


def _normalized_column(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _find_column(fieldnames: Sequence[str], aliases: Sequence[str]) -> str:
    normalized: dict[str, str] = {}
    for field in fieldnames:
        key = _normalized_column(field)
        if key and key not in normalized:
            normalized[key] = field
    for alias in aliases:
        match = normalized.get(_normalized_column(alias))
        if match is not None:
            return match
    raise ValueError(f"score CSV lacks any of the columns {tuple(aliases)}")


def _finite_float(value: object, *, field: str, source: Path, row: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{source}:{row}: {field} must be a finite real value")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source}:{row}: {field} must be a finite real value"
        ) from error
    if not math.isfinite(result):
        raise ValueError(f"{source}:{row}: {field} must be finite")
    return result


def _validate_candidate_name(name: str) -> str:
    candidate = name.strip()
    if not _CANDIDATE_PATTERN.fullmatch(candidate):
        raise ValueError(f"invalid candidate name: {name!r}")
    return candidate


def parse_named_paths(values: Iterable[str], *, field: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path_text = raw.partition("=")
        if not separator or not path_text.strip():
            raise ValueError(f"{field} must use NAME=PATH syntax: {raw!r}")
        candidate = _validate_candidate_name(name)
        if candidate in result:
            raise ValueError(f"duplicate {field} candidate: {candidate}")
        result[candidate] = Path(path_text).expanduser()
    if not result:
        raise ValueError(f"at least one {field} is required")
    return result


def load_score_csv(candidate: str, path: Path) -> dict[str, CandidateScore]:
    candidate = _validate_candidate_name(candidate)
    source = path.expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"score CSV is missing: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"score CSV has no header: {source}")
        sequence_column = _find_column(
            reader.fieldnames,
            ("sequence", "sequence_name", "seq", "name"),
        )
        hota_column = _find_column(
            reader.fieldnames,
            ("CODABENCH_HOTA", "codabench_hota", "hota_0", "hota"),
        )
        mota_column = _find_column(
            reader.fieldnames,
            ("CODABENCH_MOTA", "codabench_mota", "mota"),
        )
        idf1_column = _find_column(
            reader.fieldnames,
            ("CODABENCH_IDF1", "codabench_idf1", "idf1"),
        )
        scores: dict[str, CandidateScore] = {}
        for row_number, row in enumerate(reader, start=2):
            sequence = str(row.get(sequence_column, "")).strip()
            if not sequence:
                raise ValueError(f"{source}:{row_number}: empty sequence name")
            normalized = _normalized_column(sequence)
            if normalized in {"combinedseq", "combined", "overall", "all"}:
                continue
            if sequence in scores:
                raise ValueError(
                    f"{source}:{row_number}: duplicate sequence {sequence!r}"
                )
            scores[sequence] = CandidateScore(
                sequence=sequence,
                candidate=candidate,
                hota=_finite_float(
                    row.get(hota_column),
                    field="CODABENCH_HOTA",
                    source=source,
                    row=row_number,
                ),
                mota=_finite_float(
                    row.get(mota_column),
                    field="CODABENCH_MOTA",
                    source=source,
                    row=row_number,
                ),
                idf1=_finite_float(
                    row.get(idf1_column),
                    field="CODABENCH_IDF1",
                    source=source,
                    row=row_number,
                ),
            )
    if not scores:
        raise ValueError(f"score CSV has no sequence rows: {source}")
    return scores


def load_score_bank(
    score_paths: Mapping[str, Path],
    *,
    raw_candidate: str,
) -> dict[str, dict[str, CandidateScore]]:
    raw_candidate = _validate_candidate_name(raw_candidate)
    if raw_candidate not in score_paths:
        raise ValueError(f"raw score CSV is missing: {raw_candidate}")
    bank = {
        candidate: load_score_csv(candidate, path)
        for candidate, path in sorted(score_paths.items())
    }
    expected = set(bank[raw_candidate])
    for candidate, scores in bank.items():
        observed = set(scores)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"{candidate}: score coverage mismatch; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return bank


def build_stratified_folds(
    sequences: Sequence[str],
    *,
    fold_count: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    unique = tuple(dict.fromkeys(sequences))
    if len(unique) != len(sequences):
        raise ValueError("sequence names must be unique")
    if fold_count < 2 or fold_count > len(unique):
        raise ValueError("fold_count must be in [2, number of sequences]")
    groups: dict[str, list[str]] = defaultdict(list)
    for sequence in unique:
        groups[sequence_prefix(sequence)].append(sequence)
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    randomizer = random.Random(seed)
    for prefix in sorted(groups):
        items = sorted(groups[prefix])
        random.Random(f"{seed}:{prefix}").shuffle(items)
        offset = randomizer.randrange(fold_count)
        for index, sequence in enumerate(items):
            folds[(offset + index) % fold_count].append(sequence)
    empty_indices = [index for index, fold in enumerate(folds) if not fold]
    for empty_index in empty_indices:
        donor_index = max(range(fold_count), key=lambda index: len(folds[index]))
        if len(folds[donor_index]) <= 1:
            raise ValueError("stratification produced an empty fold")
        folds[empty_index].append(folds[donor_index].pop())
    return tuple(tuple(sorted(fold)) for fold in folds)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return sum(materialized) / len(materialized)


def _candidate_gains(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    *,
    candidate: str,
    raw_candidate: str,
    sequences: Sequence[str],
) -> dict[str, float]:
    raw_scores = bank[raw_candidate]
    candidate_scores = bank[candidate]
    return {
        "hota": _mean(
            candidate_scores[sequence].hota - raw_scores[sequence].hota
            for sequence in sequences
        ),
        "mota": _mean(
            candidate_scores[sequence].mota - raw_scores[sequence].mota
            for sequence in sequences
        ),
        "idf1": _mean(
            candidate_scores[sequence].idf1 - raw_scores[sequence].idf1
            for sequence in sequences
        ),
    }


def fit_prefix_policy(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    sequences: Sequence[str],
    config: GateConfig,
) -> tuple[dict[str, str], dict[str, object]]:
    config.validate()
    selected_sequences = tuple(sequences)
    prefixes = sorted({sequence_prefix(sequence) for sequence in selected_sequences})
    mapping: dict[str, str] = {}
    diagnostics: dict[str, object] = {}
    candidates = sorted(bank)
    for prefix in prefixes:
        prefix_sequences = tuple(
            sequence
            for sequence in selected_sequences
            if sequence_prefix(sequence) == prefix
        )
        candidate_rows: list[dict[str, object]] = []
        best_candidate = config.raw_candidate
        best_rank: tuple[float, float, float, float] | None = None
        for candidate in candidates:
            gains = _candidate_gains(
                bank,
                candidate=candidate,
                raw_candidate=config.raw_candidate,
                sequences=prefix_sequences,
            )
            sample_count = len(prefix_sequences)
            shrinkage = sample_count / (sample_count + config.prior_strength)
            shrunk_hota_gain = gains["hota"] * shrinkage
            eligible = candidate == config.raw_candidate or (
                sample_count >= config.min_prefix_samples
                and shrunk_hota_gain >= config.min_train_hota_gain
                and gains["mota"] >= -config.max_train_mota_drop
                and gains["idf1"] >= -config.max_train_idf1_drop
            )
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "sample_count": sample_count,
                    "mean_hota_gain": gains["hota"],
                    "shrunk_hota_gain": shrunk_hota_gain,
                    "mean_mota_gain": gains["mota"],
                    "mean_idf1_gain": gains["idf1"],
                    "eligible": eligible,
                }
            )
            if candidate == config.raw_candidate or not eligible:
                continue
            rank = (
                shrunk_hota_gain,
                gains["hota"],
                gains["idf1"],
                gains["mota"],
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate
            elif rank == best_rank and candidate < best_candidate:
                best_candidate = candidate
        mapping[prefix] = best_candidate
        diagnostics[prefix] = {
            "sequence_count": len(prefix_sequences),
            "selected_candidate": best_candidate,
            "candidates": candidate_rows,
        }
    return mapping, diagnostics


def _evaluate_mapping(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    mapping: Mapping[str, str],
    *,
    raw_candidate: str,
    sequences: Sequence[str],
    fold_index: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sequence in sequences:
        prefix = sequence_prefix(sequence)
        candidate = mapping.get(prefix, raw_candidate)
        selected = bank[candidate][sequence]
        raw = bank[raw_candidate][sequence]
        rows.append(
            {
                "fold": fold_index,
                "sequence": sequence,
                "prefix": prefix,
                "candidate": candidate,
                "hota": selected.hota,
                "mota": selected.mota,
                "idf1": selected.idf1,
                "raw_hota": raw.hota,
                "raw_mota": raw.mota,
                "raw_idf1": raw.idf1,
                "hota_delta": selected.hota - raw.hota,
                "mota_delta": selected.mota - raw.mota,
                "idf1_delta": selected.idf1 - raw.idf1,
            }
        )
    return rows


def cross_validate_policy(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    config: GateConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config.validate()
    sequences = tuple(sorted(bank[config.raw_candidate]))
    folds = build_stratified_folds(
        sequences,
        fold_count=config.fold_count,
        seed=config.seed,
    )
    sequence_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    all_sequences = set(sequences)
    for fold_index, held_out in enumerate(folds):
        training = tuple(sorted(all_sequences - set(held_out)))
        mapping, _diagnostics = fit_prefix_policy(bank, training, config)
        rows = _evaluate_mapping(
            bank,
            mapping,
            raw_candidate=config.raw_candidate,
            sequences=held_out,
            fold_index=fold_index,
        )
        sequence_rows.extend(rows)
        fold_rows.append(
            {
                "fold": fold_index,
                "training_sequence_count": len(training),
                "held_out_sequence_count": len(held_out),
                "mean_hota_delta": _mean(float(row["hota_delta"]) for row in rows),
                "mean_mota_delta": _mean(float(row["mota_delta"]) for row in rows),
                "mean_idf1_delta": _mean(float(row["idf1_delta"]) for row in rows),
                "policy": mapping,
            }
        )
    mean_hota_delta = _mean(float(row["hota_delta"]) for row in sequence_rows)
    mean_mota_delta = _mean(float(row["mota_delta"]) for row in sequence_rows)
    mean_idf1_delta = _mean(float(row["idf1_delta"]) for row in sequence_rows)
    prefix_deltas: dict[str, float] = {}
    for prefix in sorted({str(row["prefix"]) for row in sequence_rows}):
        prefix_deltas[prefix] = _mean(
            float(row["hota_delta"])
            for row in sequence_rows
            if row["prefix"] == prefix
        )
    worst_prefix = min(prefix_deltas, key=prefix_deltas.get)
    worst_prefix_delta = prefix_deltas[worst_prefix]
    reasons: list[str] = []
    if mean_hota_delta < config.min_cv_hota_gain:
        reasons.append("mean_cv_hota_gain")
    if mean_mota_delta < -config.max_cv_mota_drop:
        reasons.append("mean_cv_mota_drop")
    if mean_idf1_delta < -config.max_cv_idf1_drop:
        reasons.append("mean_cv_idf1_drop")
    if worst_prefix_delta < -config.max_worst_prefix_hota_drop:
        reasons.append("worst_prefix_hota_drop")
    summary = {
        "folds": [list(fold) for fold in folds],
        "fold_rows": fold_rows,
        "mean_hota_delta": mean_hota_delta,
        "mean_mota_delta": mean_mota_delta,
        "mean_idf1_delta": mean_idf1_delta,
        "prefix_hota_deltas": prefix_deltas,
        "worst_prefix": worst_prefix,
        "worst_prefix_hota_delta": worst_prefix_delta,
        "passed": not reasons,
        "rejection_reasons": reasons,
    }
    return summary, sequence_rows


def fit_guarded_policy(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    config: GateConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    config.validate()
    if config.raw_candidate not in bank:
        raise ValueError(f"raw candidate is absent: {config.raw_candidate}")
    cv_summary, cv_rows = cross_validate_policy(bank, config)
    all_sequences = tuple(sorted(bank[config.raw_candidate]))
    fitted_mapping, fit_diagnostics = fit_prefix_policy(bank, all_sequences, config)
    prefixes = sorted({sequence_prefix(sequence) for sequence in all_sequences})
    raw_fallback = not bool(cv_summary["passed"])
    if raw_fallback:
        fitted_mapping = {prefix: config.raw_candidate for prefix in prefixes}
    payload = {
        "schema": _POLICY_SCHEMA,
        "raw_candidate": config.raw_candidate,
        "prefix_to_candidate": fitted_mapping,
        "raw_fallback": raw_fallback,
        "config": asdict(config),
        "cross_validation": cv_summary,
        "fit_diagnostics": fit_diagnostics,
        "sequence_count": len(all_sequences),
        "candidates": sorted(bank),
    }
    return payload, cv_rows


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PredictionSource:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self._archive: zipfile.ZipFile | None = None
        self._names: tuple[str, ...] = ()

    def __enter__(self) -> PredictionSource:
        if self.path.is_dir():
            self._names = tuple(sorted(path.name for path in self.path.glob("*.txt")))
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            self._archive = zipfile.ZipFile(self.path, "r")
            names: list[str] = []
            seen: set[str] = set()
            for member in self._archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                if "/" in name or "\\" in name or not name.endswith(".txt"):
                    raise ValueError(f"unsafe prediction ZIP member: {name!r}")
                if name in seen:
                    raise ValueError(f"duplicate prediction ZIP member: {name!r}")
                seen.add(name)
                names.append(name)
            self._names = tuple(sorted(names))
        else:
            raise FileNotFoundError(
                f"prediction source must be a directory or ZIP: {self.path}"
            )
        if not self._names:
            raise ValueError(f"prediction source has no root-level text files: {self.path}")
        return self

    def __exit__(self, *_args: object) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def read(self, name: str) -> bytes:
        if name not in self._names:
            raise FileNotFoundError(f"{self.path}: missing prediction {name}")
        if self._archive is not None:
            return self._archive.read(name)
        return (self.path / name).read_bytes()


def _validate_policy(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or payload.get("schema") != _POLICY_SCHEMA:
        raise ValueError("unsupported scenario expert policy")
    raw_candidate = payload.get("raw_candidate")
    mapping = payload.get("prefix_to_candidate")
    if not isinstance(raw_candidate, str):
        raise ValueError("policy raw_candidate must be a string")
    _validate_candidate_name(raw_candidate)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("policy prefix_to_candidate must be a non-empty object")
    for prefix, candidate in mapping.items():
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("policy prefix names must be non-empty strings")
        if not isinstance(candidate, str):
            raise ValueError(f"policy candidate for {prefix!r} must be a string")
        _validate_candidate_name(candidate)
    return payload


def load_policy(path: Path) -> dict[str, object]:
    source = path.expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"policy JSON is missing: {source}")
    return _validate_policy(json.loads(source.read_text(encoding="utf-8")))


def _guard_output_aliases(output_dir: Path, sources: Mapping[str, Path]) -> None:
    output = output_dir.expanduser().resolve()
    for candidate, raw_path in sources.items():
        source = raw_path.expanduser().resolve()
        if source.is_dir() and (output == source or source in output.parents):
            raise ValueError(
                f"output directory must not equal or be inside {candidate}: {source}"
            )
        if source.is_file() and output == source:
            raise ValueError(f"output directory aliases {candidate}: {source}")


def _directory_digest(path: Path) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    byte_count = 0
    file_count = 0
    for prediction in sorted(path.glob("*.txt")):
        name = prediction.name.encode("utf-8")
        data = prediction.read_bytes()
        hasher.update(len(name).to_bytes(8, "big"))
        hasher.update(name)
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
        byte_count += len(data)
        file_count += 1
    return hasher.hexdigest(), byte_count, file_count


def materialize_policy(
    policy: Mapping[str, object],
    candidate_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    validated = _validate_policy(dict(policy))
    raw_candidate = str(validated["raw_candidate"])
    mapping = dict(validated["prefix_to_candidate"])
    normalized_paths = {
        _validate_candidate_name(candidate): path.expanduser()
        for candidate, path in candidate_paths.items()
    }
    required_candidates = {raw_candidate, *(str(value) for value in mapping.values())}
    missing_candidates = sorted(required_candidates - set(normalized_paths))
    if missing_candidates:
        raise ValueError(f"prediction candidates are missing: {missing_candidates}")
    _guard_output_aliases(output_dir, normalized_paths)

    selected_counts: dict[str, int] = defaultdict(int)
    with ExitStack() as stack:
        sources = {
            candidate: stack.enter_context(PredictionSource(path))
            for candidate, path in sorted(normalized_paths.items())
        }
        expected_names = set(sources[raw_candidate].names)
        for candidate, source in sources.items():
            observed = set(source.names)
            if observed != expected_names:
                missing = sorted(expected_names - observed)
                extra = sorted(observed - expected_names)
                raise ValueError(
                    f"{candidate}: prediction coverage mismatch; "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
        destination = output_dir.expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.txt"):
            if stale.name not in expected_names:
                stale.unlink()
        for name in sorted(expected_names):
            prefix = sequence_prefix(Path(name).stem)
            candidate = str(mapping.get(prefix, raw_candidate))
            payload = sources[candidate].read(name)
            _atomic_bytes(destination / name, payload)
            selected_counts[candidate] += 1
    digest, byte_count, file_count = _directory_digest(output_dir.expanduser())
    return {
        "schema": _SUMMARY_SCHEMA,
        "output_dir": str(output_dir.expanduser().resolve()),
        "file_count": file_count,
        "byte_count": byte_count,
        "sha256": digest,
        "selected_candidate_counts": dict(sorted(selected_counts.items())),
        "raw_fallback": bool(validated.get("raw_fallback", False)),
    }


def _fit_command(args: argparse.Namespace) -> int:
    score_paths = parse_named_paths(args.score_csv, field="score CSV")
    config = GateConfig(
        raw_candidate=args.raw_candidate,
        fold_count=args.fold_count,
        seed=args.seed,
        min_prefix_samples=args.min_prefix_samples,
        min_train_hota_gain=args.min_train_hota_gain,
        prior_strength=args.prior_strength,
        max_train_mota_drop=args.max_train_mota_drop,
        max_train_idf1_drop=args.max_train_idf1_drop,
        min_cv_hota_gain=args.min_cv_hota_gain,
        max_cv_mota_drop=args.max_cv_mota_drop,
        max_cv_idf1_drop=args.max_cv_idf1_drop,
        max_worst_prefix_hota_drop=args.max_worst_prefix_hota_drop,
    )
    bank = load_score_bank(score_paths, raw_candidate=config.raw_candidate)
    policy, cv_rows = fit_guarded_policy(bank, config)
    policy["score_sources"] = {
        candidate: {
            "path": str(path.expanduser().resolve()),
            "sha256": _sha256_file(path.expanduser()),
        }
        for candidate, path in sorted(score_paths.items())
    }
    output_root = args.output_dir.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    policy_path = output_root / "policy.json"
    _atomic_json(policy_path, policy)
    _write_csv(output_root / "cv_sequence_rows.csv", cv_rows)
    fold_rows = list(policy["cross_validation"]["fold_rows"])
    flattened_fold_rows = [
        {**row, "policy": json.dumps(row["policy"], sort_keys=True)}
        for row in fold_rows
    ]
    _write_csv(output_root / "cv_fold_rows.csv", flattened_fold_rows)

    materialized = None
    if args.candidate:
        candidate_paths = parse_named_paths(args.candidate, field="candidate")
        materialized = materialize_policy(
            policy,
            candidate_paths,
            output_root / "predictions",
        )
        _atomic_json(output_root / "materialization.json", materialized)
    result = {
        "schema": _SCORE_SCHEMA,
        "policy_path": str(policy_path.resolve()),
        "raw_fallback": policy["raw_fallback"],
        "cross_validation": policy["cross_validation"],
        "materialization": materialized,
    }
    _atomic_json(output_root / "fit_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_improvement and policy["raw_fallback"]:
        return 2
    return 0


def _apply_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy_json)
    candidate_paths = parse_named_paths(args.candidate, field="candidate")
    summary = materialize_policy(policy, candidate_paths, args.output_dir)
    if args.output_json is not None:
        _atomic_json(args.output_json.expanduser(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="fit and cross-validate a prefix policy")
    fit.add_argument("--score-csv", action="append", required=True, metavar="NAME=PATH")
    fit.add_argument("--candidate", action="append", default=[], metavar="NAME=PATH")
    fit.add_argument("--raw-candidate", default="raw")
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--fold-count", type=int, default=5)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--min-prefix-samples", type=int, default=3)
    fit.add_argument("--min-train-hota-gain", type=float, default=0.001)
    fit.add_argument("--prior-strength", type=float, default=3.0)
    fit.add_argument("--max-train-mota-drop", type=float, default=0.002)
    fit.add_argument("--max-train-idf1-drop", type=float, default=0.002)
    fit.add_argument("--min-cv-hota-gain", type=float, default=0.0005)
    fit.add_argument("--max-cv-mota-drop", type=float, default=0.002)
    fit.add_argument("--max-cv-idf1-drop", type=float, default=0.002)
    fit.add_argument("--max-worst-prefix-hota-drop", type=float, default=0.005)
    fit.add_argument("--require-improvement", action="store_true")
    fit.set_defaults(handler=_fit_command)

    apply = subparsers.add_parser("apply", help="materialize a fitted policy")
    apply.add_argument("--policy-json", type=Path, required=True)
    apply.add_argument("--candidate", action="append", required=True, metavar="NAME=PATH")
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--output-json", type=Path)
    apply.set_defaults(handler=_apply_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(arguments)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
