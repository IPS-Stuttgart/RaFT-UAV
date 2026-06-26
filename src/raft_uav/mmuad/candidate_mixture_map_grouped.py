"""Origin-group-aware candidate-mixture smoothing for MMUAD.

Raw and train-calibrated candidate branches can contain multiple coordinate
hypotheses for the same physical point-cloud cluster. A flat candidate softmax
implicitly gives such duplicated observations more prior mass than candidates
that have only one representation. This module removes that multiplicity bias
before delegating to the maintained robust candidate-mixture MAP smoother.

For a hypothesis group with ``n`` sibling rows, the wrapper subtracts

``group_correction_strength * log(n)``

from each sibling's mixture log weight. At strength 1 this is equivalent to
using the group's mean evidence rather than the sum of its duplicated evidence,
while retaining the normal within-group softmax preference between raw and
calibrated hypotheses. The correction is inference-safe and uses only candidate
metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
    compute_candidate_responsibilities,
    run_candidate_mixture_map,
    write_candidate_mixture_map_outputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

DEFAULT_GROUP_COLUMN_ALIASES = (
    "mmuad_calibration_origin_row",
    "candidate_origin_row",
    "origin_row",
)
MISSING_GROUP_POLICIES = ("unique", "error")
DEFAULT_CORRECTED_SCORE_COLUMN = "mixture_group_corrected_score"


@dataclass(frozen=True)
class HypothesisGroupConfig:
    """Configuration for duplicate-hypothesis multiplicity correction."""

    group_column: str | None = None
    group_column_aliases: tuple[str, ...] = DEFAULT_GROUP_COLUMN_ALIASES
    correction_strength: float = 1.0
    missing_group_policy: str = "unique"
    corrected_score_column: str = DEFAULT_CORRECTED_SCORE_COLUMN


@dataclass(frozen=True)
class GroupedCandidateMixtureMapResult:
    """Corrected candidates plus the downstream mixture-MAP result."""

    corrected_candidates: pd.DataFrame
    mixture_result: CandidateMixtureMapResult
    grouping_summary: dict[str, Any]


def prepare_hypothesis_group_candidates(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
) -> tuple[pd.DataFrame, CandidateMixtureMapConfig, dict[str, Any]]:
    """Attach group-aware score corrections and return an effective MAP config."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    _validate_group_config(group_config)

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        empty = rows.copy()
        for column, dtype in (
            ("mixture_hypothesis_group", str),
            ("mixture_hypothesis_group_size", float),
            ("mixture_group_base_normalized_score", float),
            ("mixture_group_log_size_penalty", float),
            (group_config.corrected_score_column, float),
        ):
            empty[column] = pd.Series(dtype=dtype)
        effective = replace(
            mixture_config,
            score_column=group_config.corrected_score_column,
            fallback_score_columns=(),
            score_normalization="none",
        )
        return empty, effective, _grouping_summary(
            empty,
            resolved_group_column=None,
            mixture_config=mixture_config,
            effective_config=effective,
            group_config=group_config,
        )

    rows = rows.reset_index(drop=True)
    rows["mixture_group_input_row"] = np.arange(len(rows), dtype=int)
    resolved_group_column = _resolve_group_column(rows, group_config)
    rows["mixture_hypothesis_group"] = _group_values(
        rows,
        resolved_group_column=resolved_group_column,
        missing_group_policy=group_config.missing_group_policy,
    )

    raw_score = _candidate_scores(rows, mixture_config)
    normalized_score = pd.Series(np.nan, index=rows.index, dtype=float)
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        normalized_score.loc[frame.index] = _normalize_scores(
            raw_score.loc[frame.index].to_numpy(float),
            mode=mixture_config.score_normalization,
        )
    rows["mixture_group_base_normalized_score"] = normalized_score

    group_size = rows.groupby(
        ["sequence_id", "time_s", "mixture_hypothesis_group"],
        sort=False,
    )["mixture_group_input_row"].transform("size")
    rows["mixture_hypothesis_group_size"] = group_size.astype(int)
    penalty = float(group_config.correction_strength) * np.log(
        np.maximum(group_size.to_numpy(float), 1.0)
    )
    rows["mixture_group_log_size_penalty"] = penalty

    score_weight = float(mixture_config.score_weight)
    temperature = float(mixture_config.temperature)
    if abs(score_weight) > 1.0e-12:
        adjusted = normalized_score.to_numpy(float) - (
            temperature / score_weight
        ) * penalty
        effective_score_weight = score_weight
        effective_temperature = temperature
    else:
        # Preserve the original zero score contribution while still expressing
        # the multiplicity correction through the core score term.
        adjusted = -penalty
        effective_score_weight = 1.0
        effective_temperature = 1.0

    rows[group_config.corrected_score_column] = adjusted
    effective_config = replace(
        mixture_config,
        score_column=group_config.corrected_score_column,
        fallback_score_columns=(),
        score_normalization="none",
        score_weight=effective_score_weight,
        temperature=effective_temperature,
    )
    summary = _grouping_summary(
        rows,
        resolved_group_column=resolved_group_column,
        mixture_config=mixture_config,
        effective_config=effective_config,
        group_config=group_config,
    )
    return rows, effective_config, summary


def run_grouped_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> GroupedCandidateMixtureMapResult:
    """Run robust mixture-MAP with origin-group multiplicity correction."""

    group_config = group_config or HypothesisGroupConfig()
    corrected, effective_config, grouping_summary = prepare_hypothesis_group_candidates(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
    )
    core = run_candidate_mixture_map(
        corrected,
        config=effective_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    enriched_assignments = _enrich_assignments(
        core.assignments,
        corrected,
        corrected_score_column=group_config.corrected_score_column,
    )
    enriched_estimates = _enrich_estimates(core.estimates, enriched_assignments)
    summary = dict(core.summary)
    summary["hypothesis_grouping"] = grouping_summary
    enriched_result = CandidateMixtureMapResult(
        estimates=enriched_estimates,
        assignments=enriched_assignments,
        iteration_summary=core.iteration_summary,
        summary=summary,
    )
    return GroupedCandidateMixtureMapResult(
        corrected_candidates=corrected,
        mixture_result=enriched_result,
        grouping_summary=grouping_summary,
    )


def compute_grouped_candidate_responsibilities(
    candidates: pd.DataFrame,
    state_xyz: np.ndarray,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
) -> pd.DataFrame:
    """Return single-frame responsibilities after group-size correction."""

    corrected, effective_config, _ = prepare_hypothesis_group_candidates(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
    )
    responsibilities = compute_candidate_responsibilities(
        corrected,
        state_xyz,
        config=effective_config,
    )
    group_mass = responsibilities.groupby(
        "mixture_hypothesis_group",
        dropna=False,
    )["mixture_responsibility"].transform("sum")
    responsibilities["mixture_hypothesis_group_mass"] = group_mass
    return responsibilities


def write_grouped_candidate_mixture_outputs(
    result: GroupedCandidateMixtureMapResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write corrected candidates, grouping summary, and mixture outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corrected_path = output / "mmuad_group_corrected_candidates.csv"
    grouping_path = output / "mmuad_hypothesis_group_summary.json"
    result.corrected_candidates.to_csv(corrected_path, index=False)
    grouping_path.write_text(
        json.dumps(_jsonable(result.grouping_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_candidate_mixture_map_outputs(result.mixture_result, output)
    paths["group_corrected_candidates_csv"] = corrected_path
    paths["hypothesis_group_summary_json"] = grouping_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-grouped-candidate-mixture-map",
        description=(
            "run origin-group-aware MMUAD candidate-mixture smoothing without "
            "double-counting raw/calibrated sibling hypotheses"
        ),
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )
    parser.add_argument("--hypothesis-group-column")
    parser.add_argument("--hypothesis-group-correction-strength", type=float, default=1.0)
    parser.add_argument(
        "--missing-hypothesis-group-policy",
        choices=MISSING_GROUP_POLICIES,
        default="unique",
    )
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.top_k,
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
        sigma_column=args.sigma_column,
        default_sigma_m=args.default_sigma_m,
        sigma_min_m=args.sigma_min_m,
        sigma_max_m=args.sigma_max_m,
        score_normalization=args.score_normalization,
        score_weight=args.score_weight,
        temperature=args.temperature,
        sigma_log_weight=args.sigma_log_weight,
        loss=args.loss,
        huber_delta=args.huber_delta,
        smoothness_weight=args.smoothness_weight,
        iterations=args.iterations,
        tolerance_m=args.tolerance_m,
        uniform_weight_floor=args.uniform_weight_floor,
        branch_balance=args.branch_balance,
        source_balance=args.source_balance,
        responsibility_floor=args.responsibility_floor,
        initialization=args.initialization,
    )
    group_config = HypothesisGroupConfig(
        group_column=args.hypothesis_group_column,
        correction_strength=args.hypothesis_group_correction_strength,
        missing_group_policy=args.missing_hypothesis_group_policy,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = (
        None if args.initial_estimates_csv is None else pd.read_csv(args.initial_estimates_csv)
    )
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_grouped_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_grouped_candidate_mixture_outputs(result, args.output_dir)

    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    if args.official_results_csv is not None:
        write_official_mmaud_results_csv(
            result.mixture_result.estimates,
            args.official_results_csv,
            classification=args.default_classification,
            class_map=class_map,
        )
        paths["official_results_csv"] = args.official_results_csv
    if args.official_zip is not None:
        write_official_ug2_codabench_zip(
            result.mixture_result.estimates,
            args.official_zip,
            classification=args.default_classification,
            class_map=class_map,
        )
        paths["official_zip"] = args.official_zip

    print("mmuad_grouped_candidate_mixture_map=ok")
    print(f"candidate_rows={len(result.corrected_candidates)}")
    print(
        "duplicate_hypothesis_row_count="
        f"{result.grouping_summary['duplicate_hypothesis_row_count']}"
    )
    pooled = result.mixture_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _candidate_scores(
    rows: pd.DataFrame,
    config: CandidateMixtureMapConfig,
) -> pd.Series:
    columns = (config.score_column, *config.fallback_score_columns)
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _normalize_scores(values: np.ndarray, *, mode: str) -> np.ndarray:
    score = np.asarray(values, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return score
    if mode == "rank":
        if len(score) <= 1:
            return np.full(len(score), 0.5, dtype=float)
        order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
        return order.astype(float) / float(len(score) - 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum <= minimum:
        return np.full(len(score), 0.5, dtype=float)
    return (score - minimum) / (maximum - minimum)


def _resolve_group_column(
    rows: pd.DataFrame,
    config: HypothesisGroupConfig,
) -> str | None:
    if config.group_column is not None:
        if config.group_column not in rows.columns:
            if config.missing_group_policy == "error":
                raise ValueError(
                    f"hypothesis group column {config.group_column!r} is missing"
                )
            return None
        return config.group_column
    for column in config.group_column_aliases:
        if column in rows.columns:
            return column
    if config.missing_group_policy == "error":
        raise ValueError(
            "no hypothesis group column found; tried "
            + ", ".join(config.group_column_aliases)
        )
    return None


def _group_values(
    rows: pd.DataFrame,
    *,
    resolved_group_column: str | None,
    missing_group_policy: str,
) -> pd.Series:
    unique = pd.Series(
        [f"row:{int(index)}" for index in rows["mixture_group_input_row"]],
        index=rows.index,
        dtype=object,
    )
    if resolved_group_column is None:
        return unique.astype(str)
    values = rows[resolved_group_column]
    missing = values.isna() | values.astype(str).str.strip().eq("")
    if missing.any() and missing_group_policy == "error":
        raise ValueError(
            f"hypothesis group column {resolved_group_column!r} contains missing values"
        )
    result = values.astype(str)
    result = result.where(~missing, unique)
    return result.astype(str)


def _grouping_summary(
    rows: pd.DataFrame,
    *,
    resolved_group_column: str | None,
    mixture_config: CandidateMixtureMapConfig,
    effective_config: CandidateMixtureMapConfig,
    group_config: HypothesisGroupConfig,
) -> dict[str, Any]:
    if rows.empty or "mixture_hypothesis_group_size" not in rows.columns:
        group_sizes = pd.Series(dtype=float)
        frame_group_count = pd.Series(dtype=float)
    else:
        group_sizes = pd.to_numeric(
            rows["mixture_hypothesis_group_size"],
            errors="coerce",
        )
        frame_group_count = rows.groupby(
            ["sequence_id", "time_s"],
            sort=False,
        )["mixture_hypothesis_group"].nunique()
    duplicate_rows = int((group_sizes > 1).sum()) if len(group_sizes) else 0
    duplicate_groups = 0
    if not rows.empty and "mixture_hypothesis_group" in rows.columns:
        group_table = rows.drop_duplicates(
            ["sequence_id", "time_s", "mixture_hypothesis_group"]
        )
        duplicate_groups = int(
            (
                pd.to_numeric(
                    group_table["mixture_hypothesis_group_size"],
                    errors="coerce",
                )
                > 1
            ).sum()
        )
    return {
        "resolved_group_column": resolved_group_column,
        "candidate_rows": int(len(rows)),
        "frame_count": int(len(frame_group_count)),
        "hypothesis_group_count": int(frame_group_count.sum())
        if len(frame_group_count)
        else 0,
        "hypothesis_groups_per_frame_mean": _safe_mean(frame_group_count),
        "duplicate_hypothesis_row_count": duplicate_rows,
        "duplicate_hypothesis_group_count": duplicate_groups,
        "group_size_mean": _safe_mean(group_sizes),
        "group_size_max": _safe_max(group_sizes),
        "group_config": asdict(group_config),
        "input_mixture_config": asdict(mixture_config),
        "effective_mixture_config": asdict(effective_config),
    }


def _enrich_assignments(
    assignments: pd.DataFrame,
    corrected_candidates: pd.DataFrame,
    *,
    corrected_score_column: str,
) -> pd.DataFrame:
    if assignments.empty:
        return assignments
    metadata_columns = [
        "mixture_group_input_row",
        "mixture_hypothesis_group",
        "mixture_hypothesis_group_size",
        "mixture_group_base_normalized_score",
        "mixture_group_log_size_penalty",
        corrected_score_column,
    ]
    metadata_columns = [
        column for column in metadata_columns if column in corrected_candidates.columns
    ]
    metadata = corrected_candidates[metadata_columns].rename(
        columns={"mixture_group_input_row": "candidate_input_row"}
    )
    out = assignments.merge(metadata, on="candidate_input_row", how="left")
    if "mixture_hypothesis_group" in out.columns:
        out["mixture_hypothesis_group_mass"] = out.groupby(
            ["sequence_id", "time_s", "mixture_hypothesis_group"],
            dropna=False,
        )["mixture_final_weight"].transform("sum")
    return out


def _enrich_estimates(
    estimates: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    if estimates.empty or assignments.empty:
        return estimates
    if "mixture_hypothesis_group_mass" not in assignments.columns:
        return estimates
    group_mass = assignments[
        [
            "sequence_id",
            "time_s",
            "mixture_hypothesis_group",
            "mixture_hypothesis_group_mass",
        ]
    ].drop_duplicates()
    records: list[dict[str, Any]] = []
    for (sequence_id, time_s), frame in group_mass.groupby(
        ["sequence_id", "time_s"],
        sort=False,
    ):
        mass = pd.to_numeric(
            frame["mixture_hypothesis_group_mass"],
            errors="coerce",
        ).fillna(0.0)
        values = mass.to_numpy(float)
        total = float(values.sum())
        if total > 0.0:
            values = values / total
        entropy = float(-np.sum(values * np.log(np.maximum(values, 1.0e-300))))
        records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "mixture_hypothesis_group_count": int(len(values)),
                "mixture_hypothesis_group_entropy": entropy,
                "mixture_effective_hypothesis_group_count": float(np.exp(entropy)),
                "mixture_dominant_hypothesis_group_mass": float(np.max(values)),
            }
        )
    diagnostics = pd.DataFrame.from_records(records)
    return estimates.merge(diagnostics, on=["sequence_id", "time_s"], how="left")


def _validate_group_config(config: HypothesisGroupConfig) -> None:
    if not 0.0 <= float(config.correction_strength) <= 1.0:
        raise ValueError("correction_strength must be within [0, 1]")
    if config.missing_group_policy not in MISSING_GROUP_POLICIES:
        raise ValueError(
            f"unsupported missing_group_policy {config.missing_group_policy!r}"
        )
    if not str(config.corrected_score_column).strip():
        raise ValueError("corrected_score_column must not be empty")


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(float))]
    return None if finite.empty else float(finite.mean())


def _safe_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(float))]
    return None if finite.empty else float(finite.max())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
