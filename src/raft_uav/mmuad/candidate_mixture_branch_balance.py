"""Branch-balanced candidate selection for MMUAD mixture-MAP inference.

A branch-preserving candidate pool is only useful if the final top-K stage does
not immediately collapse back to one score-dominant branch. Raw, dynamic,
source-calibrated, and merged candidate streams can have different score
scales, so this module normalizes scores within each branch and selects a
round-robin top-K before invoking the existing robust candidate-mixture MAP
smoother.

The implementation is inference-safe: it uses candidate metadata, scores, and
uncertainty only. Truth is optional and is passed solely to the existing local
diagnostic metric path.
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
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
    run_candidate_mixture_map,
    write_candidate_mixture_map_outputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns


BRANCH_SELECTION_CHOICES = ("round-robin", "global")
BRANCH_SCORE_NORMALIZATION_CHOICES = ("rank", "minmax", "none")
BRANCH_BALANCED_SCORE_COLUMN = "branch_balanced_score"


@dataclass(frozen=True)
class BranchBalanceConfig:
    """Configuration for branch-balanced per-frame candidate selection."""

    top_k: int = 20
    branch_column: str = "candidate_branch"
    fallback_branch_column: str = "source"
    score_column: str = "candidate_reservoir_grid_score"
    fallback_score_columns: Sequence[str] = (
        "branch_consensus_rank_score",
        "ranker_score",
        "confidence",
    )
    sigma_column: str = "predicted_sigma_m"
    default_sigma_m: float = 10.0
    branch_score_normalization: str = "rank"
    global_score_blend: float = 0.25
    selection_mode: str = "round-robin"


@dataclass(frozen=True)
class BranchBalancedMixtureMapResult:
    """Balanced candidates, mixture result, and combined summary."""

    balanced_candidates: pd.DataFrame
    mixture_result: CandidateMixtureMapResult
    summary: dict[str, Any]


def prepare_branch_balanced_candidates(
    candidates: pd.DataFrame,
    *,
    config: BranchBalanceConfig | None = None,
) -> pd.DataFrame:
    """Return at most ``top_k`` candidates per frame with branch diversity.

    Scores are normalized both within branch and globally. The configured blend
    retains some global ordering while preventing one branch's score scale from
    monopolizing top-K. In round-robin mode, one candidate is taken from each
    branch before a second candidate is taken from any branch.
    """

    config = config or BranchBalanceConfig()
    _validate_branch_config(config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            branch_balance_branch=pd.Series(dtype=str),
            branch_balance_raw_score=pd.Series(dtype=float),
            branch_balance_branch_score=pd.Series(dtype=float),
            branch_balance_global_score=pd.Series(dtype=float),
            branch_balanced_score=pd.Series(dtype=float),
            branch_balance_branch_rank=pd.Series(dtype=int),
            branch_balance_selection_round=pd.Series(dtype=int),
            branch_balance_selected_rank=pd.Series(dtype=int),
        )

    rows = rows.copy().reset_index(drop=True)
    rows["_branch_balance_input_row"] = np.arange(len(rows), dtype=int)
    rows["branch_balance_branch"] = _branch_values(rows, config=config)
    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        prepared = _prepare_frame(frame, config=config)
        selected = _select_frame(prepared, config=config)
        if not selected.empty:
            parts.append(selected)
    if not parts:
        return rows.iloc[0:0].drop(columns=["_branch_balance_input_row"], errors="ignore")
    output = pd.concat(parts, ignore_index=True, sort=False)
    output = output.drop(columns=["_branch_balance_input_row"], errors="ignore")
    return normalize_candidate_columns(output)


def run_branch_balanced_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    branch_config: BranchBalanceConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> BranchBalancedMixtureMapResult:
    """Run robust mixture-MAP after branch-balanced top-K selection."""

    branch_config = branch_config or BranchBalanceConfig()
    balanced = prepare_branch_balanced_candidates(candidates, config=branch_config)
    base_mixture_config = mixture_config or CandidateMixtureMapConfig()
    effective_mixture_config = replace(
        base_mixture_config,
        top_k=int(branch_config.top_k),
        score_column=BRANCH_BALANCED_SCORE_COLUMN,
        fallback_score_columns=(),
        score_normalization="none",
    )
    mixture = run_candidate_mixture_map(
        balanced,
        config=effective_mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    summary = {
        "input_candidate_rows": int(len(candidates)),
        "balanced_candidate_rows": int(len(balanced)),
        "input_frame_count": _frame_count(candidates),
        "balanced_frame_count": _frame_count(balanced),
        "branch_config": asdict(branch_config),
        "effective_mixture_config": asdict(effective_mixture_config),
        "balanced_branch_counts": _value_counts(balanced, "branch_balance_branch"),
        "balanced_source_counts": _value_counts(balanced, "source"),
        "balanced_candidates_per_frame_mean": _frame_stat(balanced, "mean"),
        "balanced_candidates_per_frame_p95": _frame_stat(balanced, "p95"),
        "mixture_summary": mixture.summary,
    }
    return BranchBalancedMixtureMapResult(
        balanced_candidates=balanced,
        mixture_result=mixture,
        summary=_jsonable(summary),
    )


def write_branch_balanced_mixture_outputs(
    result: BranchBalancedMixtureMapResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write balanced candidates plus the standard mixture-MAP artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = write_candidate_mixture_map_outputs(result.mixture_result, output)
    paths.update(
        {
            "balanced_candidates_csv": output / "mmuad_branch_balanced_candidates.csv",
            "branch_balance_summary_json": output / "mmuad_branch_balanced_mixture_summary.json",
        }
    )
    result.balanced_candidates.to_csv(paths["balanced_candidates_csv"], index=False)
    paths["branch_balance_summary_json"].write_text(
        json.dumps(_jsonable(result.summary), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-branch-balanced-mixture-map",
        description="run branch-balanced robust MMUAD candidate-mixture smoothing",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--fallback-branch-column", default="source")
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument(
        "--branch-score-normalization",
        choices=BRANCH_SCORE_NORMALIZATION_CHOICES,
        default="rank",
    )
    parser.add_argument("--global-score-blend", type=float, default=0.25)
    parser.add_argument(
        "--branch-selection-mode",
        choices=BRANCH_SELECTION_CHOICES,
        default="round-robin",
    )
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=("huber", "squared"), default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=("uncertainty-top1", "score-top1"),
        default="uncertainty-top1",
    )
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or (
        "branch_consensus_rank_score",
        "ranker_score",
        "confidence",
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = (
        None if args.initial_estimates_csv is None else pd.read_csv(args.initial_estimates_csv)
    )
    truth = None
    if args.truth_csv is not None:
        truth = load_evaluation_truth_file(args.truth_csv).rows
    branch_config = BranchBalanceConfig(
        top_k=args.top_k,
        branch_column=args.branch_column,
        fallback_branch_column=args.fallback_branch_column,
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
        sigma_column=args.sigma_column,
        default_sigma_m=args.default_sigma_m,
        branch_score_normalization=args.branch_score_normalization,
        global_score_blend=args.global_score_blend,
        selection_mode=args.branch_selection_mode,
    )
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.top_k,
        sigma_column=args.sigma_column,
        default_sigma_m=args.default_sigma_m,
        sigma_min_m=args.sigma_min_m,
        sigma_max_m=args.sigma_max_m,
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
    result = run_branch_balanced_candidate_mixture_map(
        candidates,
        branch_config=branch_config,
        mixture_config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_branch_balanced_mixture_outputs(result, args.output_dir)
    print("mmuad_branch_balanced_mixture_map=ok")
    print(f"balanced_candidate_rows={len(result.balanced_candidates)}")
    pooled = result.mixture_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _prepare_frame(frame: pd.DataFrame, *, config: BranchBalanceConfig) -> pd.DataFrame:
    group = frame.copy()
    group["branch_balance_raw_score"] = _candidate_scores(group, config=config)
    group["branch_balance_sigma_m"] = _candidate_sigmas(group, config=config)
    global_score = _normalize_scores(
        group["branch_balance_raw_score"],
        mode=config.branch_score_normalization,
    )
    group["branch_balance_global_score"] = global_score
    branch_score = pd.Series(0.0, index=group.index, dtype=float)
    for _, branch_rows in group.groupby("branch_balance_branch", sort=False, dropna=False):
        branch_score.loc[branch_rows.index] = _normalize_scores(
            branch_rows["branch_balance_raw_score"],
            mode=config.branch_score_normalization,
        )
    group["branch_balance_branch_score"] = branch_score
    blend = float(config.global_score_blend)
    group[BRANCH_BALANCED_SCORE_COLUMN] = (
        (1.0 - blend) * group["branch_balance_branch_score"]
        + blend * group["branch_balance_global_score"]
    )
    group["branch_balance_frame_branch_count"] = int(
        group["branch_balance_branch"].nunique(dropna=False)
    )
    return group


def _select_frame(frame: pd.DataFrame, *, config: BranchBalanceConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if config.selection_mode == "global":
        selected = _sort_candidates(frame).head(int(config.top_k)).copy()
        selected["branch_balance_branch_rank"] = selected.groupby(
            "branch_balance_branch",
            sort=False,
        ).cumcount() + 1
        selected["branch_balance_selection_round"] = selected["branch_balance_branch_rank"]
    else:
        selected = _round_robin_select(frame, top_k=int(config.top_k))
    selected = selected.reset_index(drop=True)
    selected["branch_balance_selected_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    return selected


def _round_robin_select(frame: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    branch_groups: dict[str, pd.DataFrame] = {}
    branch_keys: list[tuple[float, float, str]] = []
    for branch, group in frame.groupby("branch_balance_branch", sort=False, dropna=False):
        ordered = _sort_candidates(group).copy()
        ordered["branch_balance_branch_rank"] = np.arange(1, len(ordered) + 1, dtype=int)
        branch_text = str(branch)
        branch_groups[branch_text] = ordered
        first = ordered.iloc[0]
        branch_keys.append(
            (
                -float(first[BRANCH_BALANCED_SCORE_COLUMN]),
                -float(first["branch_balance_raw_score"]),
                branch_text,
            )
        )
    branch_order = [branch for _, _, branch in sorted(branch_keys)]
    selected_rows: list[pd.DataFrame] = []
    selected_count = 0
    round_index = 1
    while selected_count < int(top_k):
        made_progress = False
        for branch in branch_order:
            group = branch_groups[branch]
            if round_index > len(group):
                continue
            row = group.iloc[[round_index - 1]].copy()
            row["branch_balance_selection_round"] = int(round_index)
            selected_rows.append(row)
            selected_count += 1
            made_progress = True
            if selected_count >= int(top_k):
                break
        if not made_progress:
            break
        round_index += 1
    if not selected_rows:
        return frame.iloc[0:0].copy()
    return pd.concat(selected_rows, ignore_index=False, sort=False)


def _sort_candidates(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values(
        [
            BRANCH_BALANCED_SCORE_COLUMN,
            "branch_balance_raw_score",
            "branch_balance_sigma_m",
            "_branch_balance_input_row",
        ],
        ascending=[False, False, True, True],
    )


def _branch_values(rows: pd.DataFrame, *, config: BranchBalanceConfig) -> pd.Series:
    if config.branch_column in rows.columns:
        branch = rows[config.branch_column]
    elif config.fallback_branch_column in rows.columns:
        branch = rows[config.fallback_branch_column]
    else:
        branch = pd.Series("candidate", index=rows.index, dtype=object)
    text = branch.where(branch.notna(), "candidate").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, "candidate")


def _candidate_scores(rows: pd.DataFrame, *, config: BranchBalanceConfig) -> pd.Series:
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in (config.score_column, *config.fallback_score_columns):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _candidate_sigmas(rows: pd.DataFrame, *, config: BranchBalanceConfig) -> pd.Series:
    if config.sigma_column in rows.columns:
        sigma = pd.to_numeric(rows[config.sigma_column], errors="coerce")
    else:
        sigma = pd.Series(np.nan, index=rows.index, dtype=float)
    if "std_xy_m" in rows.columns:
        sigma = sigma.where(
            sigma.notna(),
            pd.to_numeric(rows["std_xy_m"], errors="coerce"),
        )
    sigma = sigma.fillna(float(config.default_sigma_m))
    return sigma.where(sigma > 0.0, float(config.default_sigma_m)).astype(float)


def _normalize_scores(values: pd.Series, *, mode: str) -> np.ndarray:
    score = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(float)
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


def _validate_branch_config(config: BranchBalanceConfig) -> None:
    if int(config.top_k) <= 0:
        raise ValueError("top_k must be positive")
    if config.branch_score_normalization not in BRANCH_SCORE_NORMALIZATION_CHOICES:
        raise ValueError(
            f"unsupported branch score normalization {config.branch_score_normalization!r}"
        )
    if config.selection_mode not in BRANCH_SELECTION_CHOICES:
        raise ValueError(f"unsupported branch selection mode {config.selection_mode!r}")
    if not 0.0 <= float(config.global_score_blend) <= 1.0:
        raise ValueError("global_score_blend must be in [0, 1]")
    if float(config.default_sigma_m) <= 0.0:
        raise ValueError("default_sigma_m must be positive")


def _frame_count(rows: pd.DataFrame) -> int:
    frame = pd.DataFrame(rows)
    if frame.empty or not {"sequence_id", "time_s"}.issubset(frame.columns):
        return 0
    return int(frame.groupby(["sequence_id", "time_s"], dropna=False).ngroups)


def _frame_stat(rows: pd.DataFrame, statistic: str) -> float:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return 0.0
    counts = frame.groupby(["sequence_id", "time_s"], dropna=False).size()
    if statistic == "p95":
        return float(counts.quantile(0.95))
    return float(counts.mean())


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    frame = pd.DataFrame(rows)
    if frame.empty or column not in frame.columns:
        return {}
    counts = frame[column].fillna("").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


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
