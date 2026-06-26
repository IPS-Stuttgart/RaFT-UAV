"""Branch/source-stratified top-K selection for MMUAD mixture-MAP.

A global score-only top-K can remove an entire raw, dynamic, calibrated, or
merged branch even though the upstream reservoir deliberately preserved it.
This module reserves a small quota per candidate branch and sensor source,
fills the remaining budget globally, and then runs the existing uncertainty-
aware mixture-MAP implementation unchanged.

The selector is inference-safe: it uses candidate metadata, scores, and learned
uncertainty only. Truth is optional and is passed solely to the existing local
diagnostic score path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
    run_candidate_mixture_map,
    write_candidate_mixture_map_outputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns


@dataclass(frozen=True)
class StratifiedMixtureTopKConfig:
    """Configuration for the final branch/source-aware mixture top-K cut."""

    top_k: int = 20
    min_per_branch: int = 1
    min_per_source: int = 1
    min_per_source_branch: int = 0
    score_column: str = "candidate_reservoir_grid_score"
    fallback_score_columns: tuple[str, ...] = ("ranker_score", "confidence")
    branch_column: str = "candidate_branch"
    sigma_column: str = "predicted_sigma_m"


@dataclass(frozen=True)
class StratifiedCandidateMixtureMapResult:
    """Selected top-K rows plus the downstream candidate-mixture result."""

    selected_candidates: pd.DataFrame
    mixture_result: CandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_stratified_mixture_candidates(
    candidates: pd.DataFrame,
    *,
    config: StratifiedMixtureTopKConfig | None = None,
) -> pd.DataFrame:
    """Return at most ``top_k`` candidates per frame while preserving diversity."""

    config = config or StratifiedMixtureTopKConfig()
    _validate_config(config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            mixture_stratified_score=pd.Series(dtype=float),
            mixture_stratified_rank=pd.Series(dtype=float),
            mixture_stratified_reason=pd.Series(dtype=str),
        )

    rows = rows.copy().reset_index(drop=True)
    branch_column = _resolve_branch_column(rows, config.branch_column)
    rows["candidate_branch"] = _branch_values(rows, branch_column)
    rows["source"] = rows["source"].fillna("candidate").astype(str)
    rows["_stratified_input_row"] = np.arange(len(rows), dtype=int)
    rows["_stratified_score"] = _candidate_score(rows, config=config)
    rows["_stratified_sigma"] = _candidate_sigma(rows, config=config)

    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        selected, reasons = _select_frame(frame, config=config)
        selected = _ranked_frame(selected).copy()
        selected["mixture_stratified_score"] = selected["_stratified_score"].to_numpy(
            float
        )
        selected["mixture_stratified_rank"] = np.arange(
            1,
            len(selected) + 1,
            dtype=float,
        )
        selected["mixture_stratified_reason"] = [
            ";".join(sorted(reasons.get(int(row_id), {"global_fill"})))
            for row_id in selected["_stratified_input_row"].astype(int)
        ]
        selected["mixture_stratified_input_count"] = int(len(frame))
        selected["mixture_stratified_input_branch_count"] = int(
            frame["candidate_branch"].astype(str).nunique()
        )
        selected["mixture_stratified_input_source_count"] = int(
            frame["source"].astype(str).nunique()
        )
        parts.append(selected)

    out = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    out = out.drop(
        columns=["_stratified_input_row", "_stratified_score", "_stratified_sigma"],
        errors="ignore",
    )
    return out.sort_values(
        ["sequence_id", "time_s", "mixture_stratified_rank"]
    ).reset_index(drop=True)


def run_stratified_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    stratified_config: StratifiedMixtureTopKConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> StratifiedCandidateMixtureMapResult:
    """Apply stratified top-K selection and run robust candidate-mixture MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    stratified_config = stratified_config or StratifiedMixtureTopKConfig(
        top_k=int(mixture_config.top_k),
        score_column=str(mixture_config.score_column),
        fallback_score_columns=tuple(mixture_config.fallback_score_columns),
        sigma_column=str(mixture_config.sigma_column),
    )
    selected = select_stratified_mixture_candidates(
        candidates,
        config=stratified_config,
    )
    effective_mixture_config = replace(
        mixture_config,
        top_k=int(stratified_config.top_k),
    )
    mixture_result = run_candidate_mixture_map(
        selected,
        config=effective_mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    summary = build_stratified_selection_summary(
        candidates,
        selected,
        stratified_config=stratified_config,
        mixture_config=effective_mixture_config,
    )
    return StratifiedCandidateMixtureMapResult(
        selected_candidates=selected,
        mixture_result=mixture_result,
        selection_summary=summary,
    )


def build_stratified_selection_summary(
    input_candidates: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    *,
    stratified_config: StratifiedMixtureTopKConfig,
    mixture_config: CandidateMixtureMapConfig,
) -> dict[str, Any]:
    """Return compact coverage diagnostics for the stratified top-K cut."""

    input_rows = normalize_candidate_columns(pd.DataFrame(input_candidates).copy())
    selected_rows = normalize_candidate_columns(
        pd.DataFrame(selected_candidates).copy()
    )
    input_counts = _frame_counts(input_rows)
    selected_counts = _frame_counts(selected_rows)
    branch_coverage: list[float] = []
    source_coverage: list[float] = []
    fully_preserved_branches = 0
    fully_preserved_sources = 0
    frame_count = 0
    selected_by_frame = {
        (str(sequence_id), float(time_s)): group
        for (sequence_id, time_s), group in selected_rows.groupby(
            ["sequence_id", "time_s"],
            sort=False,
        )
    }

    if not input_rows.empty:
        branch_column = _resolve_branch_column(
            input_rows,
            stratified_config.branch_column,
        )
        input_rows = input_rows.copy()
        input_rows["candidate_branch"] = _branch_values(input_rows, branch_column)
        for (sequence_id, time_s), group in input_rows.groupby(
            ["sequence_id", "time_s"],
            sort=False,
        ):
            frame_count += 1
            selected = selected_by_frame.get((str(sequence_id), float(time_s)))
            selected = selected if selected is not None else pd.DataFrame()
            input_branches = set(
                group["candidate_branch"].fillna("candidate").astype(str)
            )
            selected_branches = (
                set(selected["candidate_branch"].fillna("candidate").astype(str))
                if not selected.empty and "candidate_branch" in selected.columns
                else set()
            )
            input_sources = set(group["source"].fillna("candidate").astype(str))
            selected_sources = (
                set(selected["source"].fillna("candidate").astype(str))
                if not selected.empty and "source" in selected.columns
                else set()
            )
            branch_fraction = _coverage_fraction(input_branches, selected_branches)
            source_fraction = _coverage_fraction(input_sources, selected_sources)
            branch_coverage.append(branch_fraction)
            source_coverage.append(source_fraction)
            fully_preserved_branches += int(branch_fraction >= 1.0)
            fully_preserved_sources += int(source_fraction >= 1.0)

    return {
        "input_candidate_rows": int(len(input_rows)),
        "selected_candidate_rows": int(len(selected_rows)),
        "frame_count": int(frame_count),
        "input_candidates_per_frame_mean": _safe_mean(input_counts),
        "selected_candidates_per_frame_mean": _safe_mean(selected_counts),
        "selected_candidates_per_frame_max": _safe_max(selected_counts),
        "mean_branch_coverage_fraction": _safe_array_mean(branch_coverage),
        "mean_source_coverage_fraction": _safe_array_mean(source_coverage),
        "frames_all_branches_preserved": int(fully_preserved_branches),
        "frames_all_sources_preserved": int(fully_preserved_sources),
        "frames_all_branches_preserved_fraction": (
            float(fully_preserved_branches / frame_count) if frame_count else 0.0
        ),
        "frames_all_sources_preserved_fraction": (
            float(fully_preserved_sources / frame_count) if frame_count else 0.0
        ),
        "selection_reason_counts": _reason_counts(selected_rows),
        "candidate_branch_counts": _value_counts(selected_rows, "candidate_branch"),
        "source_counts": _value_counts(selected_rows, "source"),
        "stratified_config": asdict(stratified_config),
        "mixture_config": asdict(mixture_config),
    }


def write_stratified_candidate_mixture_outputs(
    result: StratifiedCandidateMixtureMapResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write selected candidates, coverage summary, and mixture outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_stratified_mixture_candidates.csv"
    summary_path = output / "mmuad_stratified_mixture_selection_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_candidate_mixture_map_outputs(result.mixture_result, output)
    paths["stratified_candidates_csv"] = selected_path
    paths["stratified_selection_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-stratified-candidate-mixture-map",
        description=(
            "run branch/source-stratified robust MMUAD candidate-mixture smoothing"
        ),
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--min-per-source-branch", type=int, default=0)
    parser.add_argument("--branch-column", default="candidate_branch")
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
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or (
        "ranker_score",
        "confidence",
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = (
        None
        if args.initial_estimates_csv is None
        else pd.read_csv(args.initial_estimates_csv)
    )
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
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
        initialization=args.initialization,
    )
    stratified_config = StratifiedMixtureTopKConfig(
        top_k=args.top_k,
        min_per_branch=args.min_per_branch,
        min_per_source=args.min_per_source,
        min_per_source_branch=args.min_per_source_branch,
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
        branch_column=args.branch_column,
        sigma_column=args.sigma_column,
    )
    result = run_stratified_candidate_mixture_map(
        candidates,
        stratified_config=stratified_config,
        mixture_config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_stratified_candidate_mixture_outputs(result, args.output_dir)
    print("mmuad_stratified_candidate_mixture_map=ok")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    print(f"estimate_rows={len(result.mixture_result.estimates)}")
    pooled = result.mixture_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _select_frame(
    frame: pd.DataFrame,
    *,
    config: StratifiedMixtureTopKConfig,
) -> tuple[pd.DataFrame, dict[int, set[str]]]:
    budget = min(int(config.top_k), len(frame))
    selected_ids: set[int] = set()
    reasons: dict[int, set[str]] = {}
    _apply_group_quota(
        frame,
        group_columns=("candidate_branch",),
        count=int(config.min_per_branch),
        reason_prefix="branch",
        budget=budget,
        selected_ids=selected_ids,
        reasons=reasons,
    )
    _apply_group_quota(
        frame,
        group_columns=("source",),
        count=int(config.min_per_source),
        reason_prefix="source",
        budget=budget,
        selected_ids=selected_ids,
        reasons=reasons,
    )
    _apply_group_quota(
        frame,
        group_columns=("source", "candidate_branch"),
        count=int(config.min_per_source_branch),
        reason_prefix="source_branch",
        budget=budget,
        selected_ids=selected_ids,
        reasons=reasons,
    )
    for row_id in _ranked_frame(frame)["_stratified_input_row"].astype(int):
        if len(selected_ids) >= budget:
            break
        if int(row_id) not in selected_ids:
            selected_ids.add(int(row_id))
            reasons.setdefault(int(row_id), set()).add("global_fill")
    selected_mask = frame["_stratified_input_row"].astype(int).isin(selected_ids)
    return frame.loc[selected_mask].copy(), reasons


def _apply_group_quota(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    count: int,
    reason_prefix: str,
    budget: int,
    selected_ids: set[int],
    reasons: dict[int, set[str]],
) -> None:
    if count <= 0 or len(selected_ids) >= budget:
        return
    group_key: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    groups = list(frame.groupby(group_key, sort=False, dropna=False))
    for quota_round in range(int(count)):
        proposals: list[tuple[float, float, str, int]] = []
        for key, group in groups:
            group_ids = set(group["_stratified_input_row"].astype(int))
            if len(group_ids & selected_ids) > quota_round:
                continue
            available = group.loc[
                ~group["_stratified_input_row"].astype(int).isin(selected_ids)
            ]
            if available.empty:
                continue
            candidate = _ranked_frame(available).iloc[0]
            proposals.append(
                (
                    -float(candidate["_stratified_score"]),
                    float(candidate["_stratified_sigma"]),
                    _group_key_text(key),
                    int(candidate["_stratified_input_row"]),
                )
            )
        proposals.sort()
        for _, _, key_text, row_id in proposals:
            if len(selected_ids) >= budget:
                return
            if row_id not in selected_ids:
                selected_ids.add(row_id)
                reasons.setdefault(row_id, set()).add(
                    f"{reason_prefix}:{key_text}"
                )


def _ranked_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["_stratified_score", "_stratified_sigma", "_stratified_input_row"],
        ascending=[False, True, True],
    )


def _candidate_score(
    rows: pd.DataFrame,
    *,
    config: StratifiedMixtureTopKConfig,
) -> pd.Series:
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in (config.score_column, *config.fallback_score_columns):
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _candidate_sigma(
    rows: pd.DataFrame,
    *,
    config: StratifiedMixtureTopKConfig,
) -> pd.Series:
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in (config.sigma_column, "std_xy_m", "std_z_m"):
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            result = result.where(result.notna(), values)
    return result.where(result > 0.0, np.inf).fillna(np.inf).astype(float)


def _resolve_branch_column(rows: pd.DataFrame, requested: str) -> str | None:
    if requested in rows.columns:
        return requested
    for column in (
        "candidate_branch",
        "mmuad_source_calibration_branch",
        "branch",
    ):
        if column in rows.columns:
            return column
    return None


def _branch_values(rows: pd.DataFrame, branch_column: str | None) -> pd.Series:
    if branch_column is None:
        values = rows.get("source", pd.Series("candidate", index=rows.index))
    else:
        values = rows[branch_column]
    text = values.where(values.notna(), "candidate").astype(str).str.strip()
    return text.where(text.str.len() > 0, "candidate")


def _group_key_text(value: object) -> str:
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    return str(value)


def _validate_config(config: StratifiedMixtureTopKConfig) -> None:
    if int(config.top_k) <= 0:
        raise ValueError("top_k must be positive")
    for name, value in (
        ("min_per_branch", config.min_per_branch),
        ("min_per_source", config.min_per_source),
        ("min_per_source_branch", config.min_per_source_branch),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative")


def _coverage_fraction(expected: set[str], retained: set[str]) -> float:
    return float(len(expected & retained) / len(expected)) if expected else 1.0


def _frame_counts(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=int)
    return rows.groupby(["sequence_id", "time_s"], dropna=False).size()


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _safe_max(values: pd.Series) -> int:
    return int(values.max()) if not values.empty else 0


def _safe_array_mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else 0.0


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in rows[column].value_counts(dropna=False).items()
    }


def _reason_counts(rows: pd.DataFrame) -> dict[str, int]:
    if "mixture_stratified_reason" not in rows.columns:
        return {}
    counts: dict[str, int] = {}
    for value in rows["mixture_stratified_reason"].dropna().astype(str):
        for reason in value.replace(",", ";").split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
