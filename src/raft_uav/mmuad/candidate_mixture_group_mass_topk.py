"""Posterior-mass adaptive hypothesis-group selection for MMUAD mixture-MAP.

Fixed group top-K spends the same budget on confident and ambiguous frames.
This module forms a state-independent posterior from the maintained
score/uncertainty unary and keeps the smallest bounded group count whose
cumulative mass reaches a train-selectable target.  The maintained spatial
selector then chooses that many distinct groups before grouped robust
mixture-MAP.  Truth is optional and is only used for downstream metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    _build_group_table,
    _candidate_unary_utility,
    select_spatial_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_topk import GROUP_SCORE_MODES
from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
)
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    GroupedCandidateMixtureMapResult,
    HypothesisGroupConfig,
    prepare_hypothesis_group_candidates,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns


@dataclass(frozen=True)
class PosteriorMassGroupTopKConfig:
    """Configuration for posterior-mass adaptive group selection."""

    min_group_top_k: int = 3
    max_group_top_k: int = 20
    target_posterior_mass: float = 0.95
    posterior_temperature: float = 1.0
    uniform_posterior_blend: float = 0.02
    max_siblings_per_group: int = 2
    group_score_mode: str = "max"
    diversity_weight: float = 0.5
    diversity_scale_m: float = 5.0
    diversity_cap_m: float = 30.0


@dataclass(frozen=True)
class PosteriorMassGroupTopKCandidateMixtureResult:
    selected_candidates: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select an adaptive number of physical hypothesis groups per frame."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    _validate_selection_config(selection_config)

    original = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(
        drop=True
    )
    enabled = int(selection_config.max_group_top_k) > 0
    if original.empty or not enabled:
        selected = original.copy()
        selected["mixture_mass_group_topk_selected"] = False
        return selected, _selection_summary(
            original,
            selected,
            selection_config=selection_config,
            enabled=enabled,
            frame_summaries=_empty_frame_summaries(),
        )

    prepared, _, grouping_summary = prepare_hypothesis_group_candidates(
        original,
        mixture_config=mixture_config,
        group_config=group_config,
    )
    prepared = prepared.copy()
    prepared["mixture_mass_group_candidate_utility"] = _candidate_unary_utility(
        prepared,
        mixture_config=mixture_config,
    )
    prepared["mixture_spatial_group_candidate_utility"] = prepared[
        "mixture_mass_group_candidate_utility"
    ]

    selected_frames: list[pd.DataFrame] = []
    frame_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), prepared_frame in prepared.groupby(
        ["sequence_id", "time_s"], sort=True, dropna=False
    ):
        groups = _build_group_table(
            prepared_frame,
            score_mode=selection_config.group_score_mode,
        )
        budget = _posterior_mass_budget(groups, selection_config=selection_config)
        input_rows = pd.to_numeric(
            prepared_frame["mixture_group_input_row"], errors="raise"
        ).astype(int)
        original_frame = original.iloc[input_rows.to_numpy()].copy().reset_index(drop=True)
        spatial_config = SpatialHypothesisGroupTopKConfig(
            group_top_k=int(budget["selected_group_budget"]),
            max_siblings_per_group=int(selection_config.max_siblings_per_group),
            group_score_mode=str(selection_config.group_score_mode),
            diversity_weight=float(selection_config.diversity_weight),
            diversity_scale_m=float(selection_config.diversity_scale_m),
            diversity_cap_m=float(selection_config.diversity_cap_m),
        )
        selected_frame, _ = select_spatial_hypothesis_group_topk(
            original_frame,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=spatial_config,
        )
        diagnostics = {
            "mixture_mass_group_topk_selected": True,
            "mixture_mass_group_budget": int(budget["selected_group_budget"]),
            "mixture_mass_group_available_groups": int(budget["available_groups"]),
            "mixture_mass_group_target_posterior_mass": float(
                selection_config.target_posterior_mass
            ),
            "mixture_mass_group_retained_posterior_mass": float(
                budget["retained_posterior_mass"]
            ),
            "mixture_mass_group_top1_posterior": float(budget["top1_posterior"]),
            "mixture_mass_group_normalized_entropy": float(
                budget["normalized_entropy"]
            ),
            "mixture_mass_group_effective_count": float(budget["effective_count"]),
        }
        for column, value in diagnostics.items():
            selected_frame[column] = value
        selected_frames.append(selected_frame)
        frame_records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_rows": int(len(prepared_frame)),
                "selected_rows": int(len(selected_frame)),
                **budget,
            }
        )

    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        [
            "sequence_id",
            "time_s",
            "mixture_spatial_group_rank",
            "mixture_spatial_group_sibling_rank",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    frame_summaries = pd.DataFrame.from_records(frame_records)
    summary = _selection_summary(
        original,
        selected,
        selection_config=selection_config,
        enabled=True,
        frame_summaries=frame_summaries,
    )
    summary["hypothesis_grouping"] = grouping_summary
    return selected, summary


def run_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> PosteriorMassGroupTopKCandidateMixtureResult:
    """Run adaptive group selection and grouped robust mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    selected, summary = select_posterior_mass_hypothesis_group_topk(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    effective_config = mixture_config
    if int(selection_config.max_group_top_k) > 0:
        effective_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    return PosteriorMassGroupTopKCandidateMixtureResult(selected, grouped, summary)


def write_posterior_mass_group_topk_outputs(
    result: PosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_posterior_mass_group_topk_candidates.csv"
    summary_path = output / "mmuad_posterior_mass_group_topk_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2), encoding="utf-8"
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["posterior_mass_group_topk_candidates_csv"] = selected_path
    paths["posterior_mass_group_topk_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-posterior-mass-group-topk",
        description="adapt MMUAD group top-K to framewise posterior mass",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--min-group-top-k", type=int, default=3)
    parser.add_argument("--max-group-top-k", type=int, default=20)
    parser.add_argument("--target-posterior-mass", type=float, default=0.95)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--uniform-posterior-blend", type=float, default=0.02)
    parser.add_argument("--max-siblings-per-group", type=int, default=2)
    parser.add_argument("--group-score-mode", choices=GROUP_SCORE_MODES, default="max")
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--diversity-scale-m", type=float, default=5.0)
    parser.add_argument("--diversity-cap-m", type=float, default=30.0)
    parser.add_argument("--row-top-k-when-disabled", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization", choices=SCORE_NORMALIZATION_CHOICES, default="minmax"
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization", choices=INITIALIZATION_CHOICES, default="uncertainty-top1"
    )
    parser.add_argument("--hypothesis-group-column")
    parser.add_argument("--hypothesis-group-correction-strength", type=float, default=1.0)
    parser.add_argument(
        "--missing-hypothesis-group-policy", choices=("unique", "error"), default="unique"
    )
    args = parser.parse_args(argv)

    fallback = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.row_top_k_when_disabled,
        score_column=args.score_column,
        fallback_score_columns=fallback,
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
        anchor_weight=args.anchor_weight,
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
    selection_config = PosteriorMassGroupTopKConfig(
        min_group_top_k=args.min_group_top_k,
        max_group_top_k=args.max_group_top_k,
        target_posterior_mass=args.target_posterior_mass,
        posterior_temperature=args.posterior_temperature,
        uniform_posterior_blend=args.uniform_posterior_blend,
        max_siblings_per_group=args.max_siblings_per_group,
        group_score_mode=args.group_score_mode,
        diversity_weight=args.diversity_weight,
        diversity_scale_m=args.diversity_scale_m,
        diversity_cap_m=args.diversity_cap_m,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial = (
        None
        if args.initial_estimates_csv is None
        else read_estimate_csv(args.initial_estimates_csv)
    )
    truth = (
        None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        initial_estimates=initial,
        truth=truth,
    )
    paths = write_posterior_mass_group_topk_outputs(result, args.output_dir)
    print("mmuad_candidate_mixture_posterior_mass_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    print(
        "selected_group_budget_mean="
        f"{result.selection_summary.get('selected_group_budget_mean')}"
    )
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled", {}
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _posterior_mass_budget(
    groups: pd.DataFrame,
    *,
    selection_config: PosteriorMassGroupTopKConfig,
) -> dict[str, Any]:
    count = int(len(groups))
    if count == 0:
        return {
            "available_groups": 0,
            "selected_group_budget": 0,
            "retained_posterior_mass": 0.0,
            "top1_posterior": float("nan"),
            "normalized_entropy": float("nan"),
            "effective_count": 0.0,
        }
    logits = pd.to_numeric(
        groups["mixture_spatial_group_score"], errors="coerce"
    ).to_numpy(float)
    probabilities = _softmax_probabilities(
        logits, temperature=selection_config.posterior_temperature
    )
    blend = float(selection_config.uniform_posterior_blend)
    probabilities = (1.0 - blend) * probabilities + blend / count
    sorted_probabilities = np.sort(probabilities)[::-1]
    cumulative = np.cumsum(sorted_probabilities)
    required = int(
        np.searchsorted(
            cumulative, float(selection_config.target_posterior_mass), side="left"
        )
        + 1
    )
    lower = min(int(selection_config.min_group_top_k), count)
    upper = min(int(selection_config.max_group_top_k), count)
    budget = min(max(required, lower), upper)
    entropy = _entropy(probabilities)
    return {
        "available_groups": count,
        "selected_group_budget": int(budget),
        "retained_posterior_mass": float(cumulative[budget - 1]),
        "top1_posterior": float(sorted_probabilities[0]),
        "normalized_entropy": float(entropy / np.log(count)) if count > 1 else 0.0,
        "effective_count": float(np.exp(entropy)),
    }


def _softmax_probabilities(values: np.ndarray, *, temperature: float) -> np.ndarray:
    logits = np.asarray(values, dtype=float)
    finite = np.isfinite(logits)
    if not finite.any():
        return np.full(len(logits), 1.0 / max(len(logits), 1), dtype=float)
    floor = float(np.min(logits[finite])) - 50.0
    scaled = np.where(finite, logits, floor) / float(temperature)
    scaled -= float(np.max(scaled))
    weights = np.exp(np.clip(scaled, -700.0, 0.0))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(logits), 1.0 / max(len(logits), 1), dtype=float)
    return weights / total


def _entropy(probabilities: np.ndarray) -> float:
    positive = np.asarray(probabilities, dtype=float)
    positive = positive[positive > 0.0]
    return float(-np.sum(positive * np.log(positive))) if positive.size else 0.0


def _validate_selection_config(config: PosteriorMassGroupTopKConfig) -> None:
    minimum = int(config.min_group_top_k)
    maximum = int(config.max_group_top_k)
    if maximum == 0:
        if minimum != 0:
            raise ValueError("min_group_top_k must be zero when selection is disabled")
    elif minimum <= 0 or maximum < minimum:
        raise ValueError("require 1 <= min_group_top_k <= max_group_top_k")
    if not 0.0 < float(config.target_posterior_mass) <= 1.0:
        raise ValueError("target_posterior_mass must be in (0, 1]")
    if float(config.posterior_temperature) <= 0.0:
        raise ValueError("posterior_temperature must be positive")
    if not 0.0 <= float(config.uniform_posterior_blend) < 1.0:
        raise ValueError("uniform_posterior_blend must be in [0, 1)")
    if int(config.max_siblings_per_group) < 0:
        raise ValueError("max_siblings_per_group must be non-negative")
    if config.group_score_mode not in GROUP_SCORE_MODES:
        raise ValueError(f"unsupported group_score_mode={config.group_score_mode!r}")
    if float(config.diversity_weight) < 0.0:
        raise ValueError("diversity_weight must be non-negative")
    if float(config.diversity_scale_m) <= 0.0:
        raise ValueError("diversity_scale_m must be positive")
    if float(config.diversity_cap_m) < 0.0:
        raise ValueError("diversity_cap_m must be non-negative")


def _empty_frame_summaries() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "input_rows",
            "selected_rows",
            "available_groups",
            "selected_group_budget",
            "retained_posterior_mass",
            "top1_posterior",
            "normalized_entropy",
            "effective_count",
        ]
    )


def _selection_summary(
    original: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    selection_config: PosteriorMassGroupTopKConfig,
    enabled: bool,
    frame_summaries: pd.DataFrame,
) -> dict[str, Any]:
    budgets = _numeric_column(frame_summaries, "selected_group_budget")
    retained = _numeric_column(frame_summaries, "retained_posterior_mass")
    entropy = _numeric_column(frame_summaries, "normalized_entropy")
    return {
        "schema": "raft-uav-mmuad-posterior-mass-group-topk-v1",
        "enabled": bool(enabled),
        "config": asdict(selection_config),
        "input_rows": int(len(original)),
        "selected_rows": int(len(selected)),
        "frame_count": int(
            original[["sequence_id", "time_s"]].drop_duplicates().shape[0]
        )
        if not original.empty
        else 0,
        "selected_group_budget_mean": _safe_stat(budgets, "mean"),
        "selected_group_budget_p50": _safe_stat(budgets, "quantile", 0.50),
        "selected_group_budget_p95": _safe_stat(budgets, "quantile", 0.95),
        "selected_group_budget_min": _safe_stat(budgets, "min"),
        "selected_group_budget_max": _safe_stat(budgets, "max"),
        "retained_posterior_mass_mean": _safe_stat(retained, "mean"),
        "normalized_entropy_mean": _safe_stat(entropy, "mean"),
        "frame_summaries": frame_summaries.to_dict(orient="records"),
        "truth_used_for_group_budget": False,
    }


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(rows[column], errors="coerce")
    return values.loc[np.isfinite(values)]


def _safe_stat(values: pd.Series, operation: str, argument: float | None = None) -> float:
    if values.empty:
        return float("nan")
    if operation == "mean":
        return float(values.mean())
    if operation == "quantile":
        return float(values.quantile(float(argument)))
    if operation == "min":
        return float(values.min())
    if operation == "max":
        return float(values.max())
    raise ValueError(f"unsupported statistic operation={operation!r}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
