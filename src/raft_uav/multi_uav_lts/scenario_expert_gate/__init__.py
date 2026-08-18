"""Hardened compatibility package for the Multi-UAV LTS scenario expert gate.

The maintained implementation lives in the sibling ``scenario_expert_gate.py``
module. This package keeps the public import and ``python -m`` paths stable
while adding metric-CSV compatibility, strict controls, family-wise confidence
gates, and atomic prediction materialization.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "scenario_expert_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._scenario_expert_gate_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load scenario expert implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

CandidateScore = _IMPL.CandidateScore
PredictionSource = _IMPL.PredictionSource
sequence_prefix = _IMPL.sequence_prefix
parse_named_paths = _IMPL.parse_named_paths
load_score_csv = _IMPL.load_score_csv
load_score_bank = _IMPL.load_score_bank
build_stratified_folds = _IMPL.build_stratified_folds
cross_validate_policy = _IMPL.cross_validate_policy
load_policy = _IMPL.load_policy

_ORIGINAL_FIND_COLUMN = _IMPL._find_column
_ORIGINAL_FIT_PREFIX_POLICY = _IMPL.fit_prefix_policy
_ORIGINAL_FIT_GUARDED_POLICY = _IMPL.fit_guarded_policy
_ORIGINAL_MATERIALIZE_POLICY = _IMPL.materialize_policy
_SCORE_SCHEMA = _IMPL._SCORE_SCHEMA


@dataclass(frozen=True)
class GateConfig:
    """Scenario-gate controls with confidence-bound selection guards."""

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
    bootstrap_samples: int = 5000
    familywise_alpha: float = 0.05
    min_train_hota_ci_low: float = 0.0
    min_cv_hota_ci_low: float = 0.0

    def validate(self) -> None:
        if not _IMPL._CANDIDATE_PATTERN.fullmatch(self.raw_candidate):
            raise ValueError("raw candidate has an invalid name")
        _integer_at_least(self.fold_count, "fold_count", 2)
        _integer(self.seed, "seed")
        _integer_at_least(self.min_prefix_samples, "min_prefix_samples", 1)
        _integer_at_least(self.bootstrap_samples, "bootstrap_samples", 0)
        _open_unit_interval(self.familywise_alpha, "familywise_alpha")
        _nonnegative(self.prior_strength, "prior_strength")
        for name in (
            "min_train_hota_gain",
            "max_train_mota_drop",
            "max_train_idf1_drop",
            "min_cv_hota_gain",
            "max_cv_mota_drop",
            "max_cv_idf1_drop",
            "max_worst_prefix_hota_drop",
        ):
            _nonnegative(getattr(self, name), name)
        _finite(self.min_train_hota_ci_low, "min_train_hota_ci_low")
        _finite(self.min_cv_hota_ci_low, "min_cv_hota_ci_low")


def _find_column(fieldnames: Sequence[str], aliases: Sequence[str]) -> str:
    """Accept the HOTA-at-0.05 column emitted by RaFT-UAV metrics."""

    normalized = {_IMPL._normalized_column(alias) for alias in aliases}
    expanded = tuple(aliases)
    if normalized & {"codabenchhota", "hota0", "hota"}:
        expanded = (*expanded, "hota_at_005", "hota_at_0_05")
    return _ORIGINAL_FIND_COLUMN(fieldnames, expanded)


def fit_prefix_policy(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    sequences: Sequence[str],
    config: GateConfig,
) -> tuple[dict[str, str], dict[str, object]]:
    """Fit the legacy shrinkage policy and enforce per-prefix confidence bounds."""

    config.validate()
    mapping, diagnostics = _ORIGINAL_FIT_PREFIX_POLICY(bank, sequences, config)
    comparison_count = max(1, len(bank) - 1)
    for prefix, payload in diagnostics.items():
        if not isinstance(payload, dict):
            continue
        prefix_sequences = tuple(
            sequence for sequence in sequences if sequence_prefix(sequence) == prefix
        )
        rows = payload.get("candidates", [])
        for row in rows:
            candidate = str(row.get("candidate", ""))
            if candidate == config.raw_candidate:
                row["hota_gain_ci_low"] = 0.0
                row["hota_gain_ci_high"] = 0.0
                continue
            deltas = tuple(
                bank[candidate][sequence].hota
                - bank[config.raw_candidate][sequence].hota
                for sequence in prefix_sequences
            )
            low, high = _bootstrap_interval(
                deltas,
                samples=config.bootstrap_samples,
                alpha=config.familywise_alpha,
                comparison_count=comparison_count,
                seed=_stable_seed(config.seed, prefix, candidate),
            )
            row["hota_gain_ci_low"] = low
            row["hota_gain_ci_high"] = high
            if bool(row.get("eligible")) and low < config.min_train_hota_ci_low:
                row["eligible"] = False
                row["rejection_reason"] = "train_hota_ci_low"

        eligible = [
            row
            for row in rows
            if row.get("candidate") != config.raw_candidate and row.get("eligible")
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda row: (
                    -float(row["shrunk_hota_gain"]),
                    -float(row["hota_gain_ci_low"]),
                    -float(row["mean_hota_gain"]),
                    -float(row["mean_idf1_gain"]),
                    -float(row["mean_mota_gain"]),
                    str(row["candidate"]),
                ),
            )["candidate"]
        else:
            selected = config.raw_candidate
        mapping[prefix] = str(selected)
        payload["selected_candidate"] = str(selected)
    return mapping, diagnostics


def fit_guarded_policy(
    bank: Mapping[str, Mapping[str, CandidateScore]],
    config: GateConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Fit and cross-validate a policy with a paired global HOTA bound."""

    config.validate()
    policy, rows = _ORIGINAL_FIT_GUARDED_POLICY(bank, config)
    policy = copy.deepcopy(policy)
    summary = policy["cross_validation"]
    deltas = tuple(float(row["hota_delta"]) for row in rows)
    low, high = _bootstrap_interval(
        deltas,
        samples=config.bootstrap_samples,
        alpha=config.familywise_alpha,
        comparison_count=1,
        seed=_stable_seed(config.seed, "cross-validation", "scenario-policy"),
    )
    summary["paired_hota_gain_ci_low"] = low
    summary["paired_hota_gain_ci_high"] = high
    if not policy["raw_fallback"] and low < config.min_cv_hota_ci_low:
        summary["passed"] = False
        summary["rejection_reasons"].append("paired_cv_hota_ci_low")
        policy["raw_fallback"] = True
        policy["prefix_to_candidate"] = {
            prefix: config.raw_candidate
            for prefix in policy["prefix_to_candidate"]
        }
    return policy, rows


def materialize_policy(
    policy: Mapping[str, object],
    candidate_paths: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, object]:
    """Publish a mixed prediction directory atomically and without input aliases."""

    final = output_dir.expanduser().resolve()
    _guard_disjoint(final, candidate_paths.values())
    if final.exists() and not final.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_name(f".{final.name}.tmp-{os.getpid()}")
    backup = final.with_name(f".{final.name}.backup-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    try:
        summary = _ORIGINAL_MATERIALIZE_POLICY(policy, candidate_paths, temporary)
        if final.exists():
            final.replace(backup)
        try:
            temporary.replace(final)
        except Exception:
            if backup.exists() and not final.exists():
                backup.replace(final)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    summary = dict(summary)
    summary["output_dir"] = str(final)
    return summary


def _guard_disjoint(output: Path, source_paths: Iterable[Path]) -> None:
    for raw_path in source_paths:
        source = Path(raw_path).expanduser().resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output directory must be disjoint from prediction inputs")


def _bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int,
    alpha: float,
    comparison_count: int,
    seed: int,
) -> tuple[float, float]:
    materialized = np.asarray(tuple(float(value) for value in values), dtype=float)
    if materialized.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(materialized))
    if samples <= 0 or materialized.size == 1:
        return mean, mean
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, materialized.size, size=(samples, materialized.size))
    means = np.mean(materialized[indices], axis=1)
    tail = alpha / (2.0 * max(1, comparison_count))
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def _stable_seed(seed: int, *parts: str) -> int:
    text = "\0".join(parts).encode("utf-8")
    return seed ^ int.from_bytes(__import__("hashlib").sha256(text).digest()[:8], "big")


def _finite(value: object, name: str) -> float:
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


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _open_unit_interval(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one")
    return number


def _integer(value: object, name: str) -> int:
    number = _finite(value, name)
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _integer_at_least(value: object, name: str, minimum: int) -> int:
    integer = _integer(value, name)
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


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
        bootstrap_samples=args.bootstrap_samples,
        familywise_alpha=args.familywise_alpha,
        min_train_hota_ci_low=args.min_train_hota_ci_low,
        min_cv_hota_ci_low=args.min_cv_hota_ci_low,
    )
    bank = load_score_bank(score_paths, raw_candidate=config.raw_candidate)
    policy, cv_rows = fit_guarded_policy(bank, config)
    policy["score_sources"] = {
        candidate: {
            "path": str(path.expanduser().resolve()),
            "sha256": _IMPL._sha256_file(path.expanduser()),
        }
        for candidate, path in sorted(score_paths.items())
    }
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    policy_path = output / "policy.json"
    _IMPL._atomic_json(policy_path, policy)
    _IMPL._write_csv(output / "cv_sequence_rows.csv", cv_rows)
    fold_rows = [
        {**row, "policy": json.dumps(row["policy"], sort_keys=True)}
        for row in policy["cross_validation"]["fold_rows"]
    ]
    _IMPL._write_csv(output / "cv_fold_rows.csv", fold_rows)
    materialized = None
    if args.candidate:
        candidate_paths = parse_named_paths(args.candidate, field="candidate")
        materialized = materialize_policy(policy, candidate_paths, output / "predictions")
        _IMPL._atomic_json(output / "materialization.json", materialized)
    result = {
        "schema": _SCORE_SCHEMA,
        "policy_path": str(policy_path.resolve()),
        "raw_fallback": policy["raw_fallback"],
        "cross_validation": policy["cross_validation"],
        "materialization": materialized,
    }
    _IMPL._atomic_json(output / "fit_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if args.require_improvement and policy["raw_fallback"] else 0


def _apply_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy_json)
    candidate_paths = parse_named_paths(args.candidate, field="candidate")
    summary = materialize_policy(policy, candidate_paths, args.output_dir)
    if args.output_json is not None:
        _IMPL._atomic_json(args.output_json.expanduser(), summary)
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
    fit.add_argument("--bootstrap-samples", type=int, default=5000)
    fit.add_argument("--familywise-alpha", type=float, default=0.05)
    fit.add_argument("--min-train-hota-ci-low", type=float, default=0.0)
    fit.add_argument("--min-cv-hota-ci-low", type=float, default=0.0)
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
    args = _build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        _build_parser().error(str(exc))


# Make legacy helper calls use the hardened boundaries as well.
_IMPL._find_column = _find_column
_IMPL.GateConfig = GateConfig
_IMPL.fit_prefix_policy = fit_prefix_policy
_IMPL.fit_guarded_policy = fit_guarded_policy
_IMPL.materialize_policy = materialize_policy

__all__ = [
    "CandidateScore",
    "GateConfig",
    "PredictionSource",
    "build_stratified_folds",
    "cross_validate_policy",
    "fit_guarded_policy",
    "fit_prefix_policy",
    "load_policy",
    "load_score_bank",
    "load_score_csv",
    "main",
    "materialize_policy",
    "parse_named_paths",
    "sequence_prefix",
]
