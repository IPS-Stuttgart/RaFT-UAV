"""Select scenario-specific LTS box variants from complementary-fold references."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .fixed_population_cv import scenario_prefix
from .metrics import BenchmarkMetrics, evaluate_lts_predictions

_SCHEMA = "raft-uav-multi-uav-lts-scenario-box-policy-v1"


@dataclass(frozen=True)
class VariantScore:
    candidate: str
    sequence_count: int
    codabench_hota: float
    codabench_idf1: float
    codabench_mota: float


@dataclass(frozen=True)
class PrefixSelection:
    fold: int
    prefix: str
    support_sequences: tuple[str, ...]
    used_global_fallback: bool
    selected_candidate: str
    scores: tuple[VariantScore, ...]


@dataclass(frozen=True)
class ScenarioPolicySummary:
    schema: str
    truth_dir: str
    output_dir: str
    fold_count: int
    candidate_names: tuple[str, ...]
    sequence_count: int
    selections: tuple[PrefixSelection, ...]


def assemble_scenario_policy_predictions(
    reference_candidates: Mapping[str, Path],
    target_candidates: Mapping[str, Path],
    truth_dir: Path,
    folds: Sequence[Sequence[str]],
    output_dir: Path,
) -> ScenarioPolicySummary:
    """Learn prefix routing on reference predictions and apply it to target outputs.

    ``reference_candidates`` should be generated without truth-dependent fitting. This
    lets the policy use truth only on the complementary training sequences. The target
    candidates may be separately out-of-fold models, provided each target prediction
    is valid for its own held-out sequence.
    """

    reference = _validate_candidate_mapping(reference_candidates, label="reference")
    target = _validate_candidate_mapping(target_candidates, label="target")
    if tuple(reference) != tuple(target):
        raise ValueError("reference and target candidate names must match exactly")
    if not truth_dir.is_dir():
        raise NotADirectoryError(truth_dir)

    normalized_folds = tuple(tuple(str(sequence) for sequence in fold) for fold in folds)
    _validate_folds(normalized_folds)
    all_sequences = tuple(sorted(sequence for fold in normalized_folds for sequence in fold))
    truth_names = {path.stem for path in truth_dir.glob("*.txt")}
    missing_truth = sorted(set(all_sequences) - truth_names)
    if missing_truth:
        raise ValueError(f"policy folds contain sequences without truth: {', '.join(missing_truth)}")

    _validate_output_separation(tuple(target.values()), output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    selections: list[PrefixSelection] = []
    copied: set[str] = set()
    all_set = set(all_sequences)
    for fold_index, heldout in enumerate(normalized_folds):
        heldout_set = set(heldout)
        training = tuple(sequence for sequence in all_sequences if sequence not in heldout_set)
        if not training:
            raise ValueError(f"fold {fold_index} has no policy-training sequences")
        heldout_prefixes = sorted({scenario_prefix(sequence) for sequence in heldout})
        selected_by_prefix: dict[str, str] = {}
        for prefix in heldout_prefixes:
            prefix_support = tuple(
                sequence for sequence in training if scenario_prefix(sequence) == prefix
            )
            support = prefix_support or training
            scores = tuple(
                _score_candidate(name, reference[name], truth_dir, support)
                for name in reference
            )
            selected = min(
                scores,
                key=lambda score: (
                    -score.codabench_hota,
                    -score.codabench_idf1,
                    -score.codabench_mota,
                    score.candidate,
                ),
            )
            selected_by_prefix[prefix] = selected.candidate
            selections.append(
                PrefixSelection(
                    fold=fold_index,
                    prefix=prefix,
                    support_sequences=support,
                    used_global_fallback=not bool(prefix_support),
                    selected_candidate=selected.candidate,
                    scores=scores,
                )
            )

        for sequence in heldout:
            candidate = selected_by_prefix[scenario_prefix(sequence)]
            source = target[candidate] / f"{sequence}.txt"
            if not source.is_file():
                raise FileNotFoundError(
                    f"target candidate {candidate!r} is missing {sequence}.txt"
                )
            destination = output_dir / source.name
            if destination.name in copied:
                raise ValueError(f"policy output duplicates sequence {destination.name}")
            shutil.copy2(source, destination)
            copied.add(destination.name)

    expected_names = {f"{sequence}.txt" for sequence in all_set}
    if copied != expected_names:
        raise ValueError(
            "policy output coverage mismatch: "
            f"missing={sorted(expected_names - copied)}, extra={sorted(copied - expected_names)}"
        )
    return ScenarioPolicySummary(
        schema=_SCHEMA,
        truth_dir=str(truth_dir),
        output_dir=str(output_dir),
        fold_count=len(normalized_folds),
        candidate_names=tuple(reference),
        sequence_count=len(all_sequences),
        selections=tuple(selections),
    )


def _score_candidate(
    name: str,
    prediction_dir: Path,
    truth_dir: Path,
    sequences: tuple[str, ...],
) -> VariantScore:
    metrics: BenchmarkMetrics = evaluate_lts_predictions(
        prediction_dir,
        truth_dir,
        sequences=sequences,
    )
    return VariantScore(
        candidate=name,
        sequence_count=metrics.sequence_count,
        codabench_hota=metrics.codabench_hota,
        codabench_idf1=metrics.codabench_idf1,
        codabench_mota=metrics.codabench_mota,
    )


def _validate_candidate_mapping(
    candidates: Mapping[str, Path],
    *,
    label: str,
) -> dict[str, Path]:
    if not candidates:
        raise ValueError(f"{label} candidate mapping must be non-empty")
    normalized: dict[str, Path] = {}
    for raw_name, raw_path in sorted(candidates.items()):
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{label} candidate names must be non-empty")
        path = Path(raw_path)
        if not path.is_dir():
            raise NotADirectoryError(path)
        normalized[name] = path
    return normalized


def _validate_folds(folds: tuple[tuple[str, ...], ...]) -> None:
    if len(folds) < 2:
        raise ValueError("scenario policy requires at least two folds")
    seen: set[str] = set()
    for index, fold in enumerate(folds):
        if not fold:
            raise ValueError(f"scenario policy fold {index} is empty")
        if len(set(fold)) != len(fold):
            raise ValueError(f"scenario policy fold {index} contains duplicates")
        overlap = seen & set(fold)
        if overlap:
            raise ValueError(f"scenario policy folds overlap: {sorted(overlap)}")
        seen.update(fold)


def _validate_output_separation(candidate_dirs: tuple[Path, ...], output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    for candidate_dir in candidate_dirs:
        source = candidate_dir.expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("policy output must not alias or be nested in a target candidate")


def write_summary(summary: ScenarioPolicySummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
