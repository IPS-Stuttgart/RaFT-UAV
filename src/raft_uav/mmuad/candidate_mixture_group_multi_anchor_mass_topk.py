"""Multi-anchor adaptive hypothesis-group selection for MMUAD mixture-MAP.

A single trajectory anchor can improve candidate-group selection, but it can also
lock the finite group budget onto the wrong local trajectory basin.  This module
conditions the existing posterior-mass group selector on several inference-time
trajectory hypotheses.  Candidate costs can be aggregated by minimum, soft
minimum, or mean anchor cost before the maintained robust grouped mixture-MAP
smoother is run.

Ground truth is optional and is never used for group selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
    add_anchor_conditioned_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
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
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file

MULTI_ANCHOR_UTILITY_COLUMN = "mixture_multi_anchor_selection_utility"
ANCHOR_AGGREGATION_CHOICES = ("min", "softmin", "mean")
FINAL_ANCHOR_POLICIES = ("none", "first", "median")
INSUFFICIENT_SUPPORT_POLICIES = ("neutral", "error")


@dataclass(frozen=True)
class MultiAnchorConditioningConfig:
    """Configuration for combining several trajectory-anchor costs."""

    aggregation: str = "softmin"
    softmin_temperature: float = 0.5
    minimum_anchor_support: int = 1
    insufficient_support_policy: str = "neutral"
    final_anchor_policy: str = "none"


@dataclass(frozen=True)
class MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Scored candidates, selected groups, and final grouped MAP result."""

    scored_candidates: pd.DataFrame
    selected_candidates: pd.DataFrame
    final_initial_estimates: pd.DataFrame | None
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def add_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    initial_estimates_by_name: Mapping[str, pd.DataFrame],
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    multi_anchor_config: MultiAnchorConditioningConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame | None, dict[str, Any]]:
    """Attach a robust aggregate of several inference-time anchor costs.

    ``aggregation="min"`` keeps a candidate cheap when any anchor supports it.
    ``aggregation="softmin"`` is a smooth any-anchor rule, while ``"mean"``
    rewards consensus among the supplied trajectories.
    """

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    multi_anchor_config = multi_anchor_config or MultiAnchorConditioningConfig()
    _validate_multi_anchor_config(multi_anchor_config)
    anchors = _validate_anchor_mapping(initial_estimates_by_name)

    scored_by_anchor: dict[str, pd.DataFrame] = {}
    normalized_anchors: dict[str, pd.DataFrame] = {}
    safe_names: dict[str, str] = {}
    used_safe_names: set[str] = set()
    base_utility: np.ndarray | None = None
    base_rows: pd.DataFrame | None = None

    for anchor_name, anchor_rows in anchors.items():
        scored, normalized, _ = add_anchor_conditioned_selection_utility(
            candidates,
            anchor_rows,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
        )
        safe_name = _unique_safe_name(anchor_name, used_safe_names)
        used_safe_names.add(safe_name)
        safe_names[anchor_name] = safe_name
        scored_by_anchor[anchor_name] = scored
        normalized_anchors[anchor_name] = normalized
        if base_rows is None:
            base_rows = scored.copy()
            base_utility = scored["mixture_anchor_base_utility"].to_numpy(float)
        else:
            _assert_candidate_alignment(base_rows, scored, anchor_name=anchor_name)
            current = scored["mixture_anchor_base_utility"].to_numpy(float)
            if not np.allclose(current, base_utility, equal_nan=True):
                raise ValueError(
                    "anchor-conditioned candidate base utilities differ across anchors"
                )

    assert base_rows is not None
    assert base_utility is not None
    out = base_rows.copy()
    cost_columns: list[str] = []
    matched_columns: list[str] = []
    anchor_coordinate_frames: list[pd.DataFrame] = []

    for anchor_name, scored in scored_by_anchor.items():
        safe_name = safe_names[anchor_name]
        cost_column = f"mixture_multi_anchor_cost__{safe_name}"
        matched_column = f"mixture_multi_anchor_matched__{safe_name}"
        out[cost_column] = pd.to_numeric(
            scored["mixture_anchor_cost"], errors="coerce"
        ).to_numpy(float)
        out[matched_column] = scored["mixture_anchor_matched"].fillna(False).astype(bool)
        cost_columns.append(cost_column)
        matched_columns.append(matched_column)
        anchor_coordinates = scored.loc[
            scored["mixture_anchor_matched"].fillna(False).astype(bool),
            [
                "sequence_id",
                "time_s",
                "mixture_anchor_x_m",
                "mixture_anchor_y_m",
                "mixture_anchor_z_m",
            ],
        ].copy()
        if not anchor_coordinates.empty:
            anchor_coordinates = anchor_coordinates.drop_duplicates(
                ["sequence_id", "time_s"], keep="first"
            )
            anchor_coordinates["anchor_name"] = anchor_name
            anchor_coordinate_frames.append(anchor_coordinates)

    costs = out[cost_columns].to_numpy(float)
    matched = out[matched_columns].to_numpy(bool)
    costs = np.where(matched, costs, np.nan)
    support_count = np.sum(matched, axis=1).astype(int)
    sufficient = support_count >= int(multi_anchor_config.minimum_anchor_support)
    if not sufficient.all() and multi_anchor_config.insufficient_support_policy == "error":
        bad = np.flatnonzero(~sufficient)
        example = ", ".join(str(int(index)) for index in bad[:5])
        raise ValueError(
            "insufficient trajectory-anchor support for candidate rows: "
            f"{example}{' ...' if len(bad) > 5 else ''}"
        )

    aggregate_cost = _aggregate_anchor_costs(
        costs,
        aggregation=multi_anchor_config.aggregation,
        temperature=float(multi_anchor_config.softmin_temperature),
    )
    aggregate_cost = np.where(sufficient, aggregate_cost, 0.0)
    cost_min = _nan_stat(costs, np.nanmin)
    cost_max = _nan_stat(costs, np.nanmax)
    cost_std = _nan_stat(costs, np.nanstd)
    best_anchor = _best_anchor_names(costs, list(anchors))

    out["mixture_multi_anchor_support_count"] = support_count
    out["mixture_multi_anchor_support_fraction"] = support_count / float(len(anchors))
    out["mixture_multi_anchor_sufficient_support"] = sufficient
    out["mixture_multi_anchor_cost"] = aggregate_cost
    out["mixture_multi_anchor_cost_min"] = cost_min
    out["mixture_multi_anchor_cost_max"] = cost_max
    out["mixture_multi_anchor_cost_std"] = cost_std
    out["mixture_multi_anchor_best_anchor"] = best_anchor
    out[MULTI_ANCHOR_UTILITY_COLUMN] = (
        base_utility - float(anchor_config.anchor_selection_weight) * aggregate_cost
    )

    final_initial_estimates = _build_final_initial_estimates(
        anchor_coordinate_frames,
        normalized_anchors,
        policy=multi_anchor_config.final_anchor_policy,
    )
    summary = _multi_anchor_summary(
        out,
        anchor_names=list(anchors),
        safe_names=safe_names,
        anchor_config=anchor_config,
        multi_anchor_config=multi_anchor_config,
        final_initial_estimates=final_initial_estimates,
    )
    return out, normalized_anchors, final_initial_estimates, summary


def select_multi_anchor_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    initial_estimates_by_name: Mapping[str, pd.DataFrame],
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    multi_anchor_config: MultiAnchorConditioningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, dict[str, Any]]:
    """Select adaptive physical groups using several trajectory hypotheses."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    multi_anchor_config = multi_anchor_config or MultiAnchorConditioningConfig()

    scored, normalized_anchors, final_anchor, anchor_summary = (
        add_multi_anchor_conditioned_selection_utility(
            candidates,
            initial_estimates_by_name,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            multi_anchor_config=multi_anchor_config,
        )
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=MULTI_ANCHOR_UTILITY_COLUMN,
        fallback_score_columns=(),
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
    )
    selected, summary = select_posterior_mass_hypothesis_group_topk(
        scored,
        mixture_config=selection_mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    summary = dict(summary)
    summary["schema"] = "raft-uav-mmuad-multi-anchor-posterior-mass-group-topk-v1"
    summary["multi_anchor_conditioning"] = anchor_summary
    summary["selection_mixture_config"] = asdict(selection_mixture_config)
    summary["normalized_anchor_rows"] = {
        name: int(len(rows)) for name, rows in normalized_anchors.items()
    }
    summary["truth_used_for_selection"] = False
    return selected, final_anchor, scored, _jsonable(summary)


def run_multi_anchor_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    initial_estimates_by_name: Mapping[str, pd.DataFrame],
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    multi_anchor_config: MultiAnchorConditioningConfig | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run multi-anchor adaptive selection followed by grouped mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    selected, final_anchor, scored, summary = (
        select_multi_anchor_posterior_mass_hypothesis_group_topk(
            candidates,
            initial_estimates_by_name=initial_estimates_by_name,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            multi_anchor_config=multi_anchor_config,
        )
    )
    effective_mixture_config = mixture_config
    if int(selection_config.max_group_top_k) > 0:
        effective_mixture_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=final_anchor,
        truth=truth,
    )
    summary["final_mixture_config"] = asdict(effective_mixture_config)
    return MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult(
        scored_candidates=scored,
        selected_candidates=selected,
        final_initial_estimates=final_anchor,
        grouped_result=grouped,
        selection_summary=_jsonable(summary),
    )


def write_multi_anchor_posterior_mass_group_topk_outputs(
    result: MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write multi-anchor diagnostics and standard grouped-mixture artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_multi_anchor_scored_candidates.csv"
    selected_path = output / "mmuad_multi_anchor_selected_candidates.csv"
    summary_path = output / "mmuad_multi_anchor_posterior_mass_group_topk_summary.json"
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2), encoding="utf-8"
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["multi_anchor_scored_candidates_csv"] = scored_path
    paths["multi_anchor_selected_candidates_csv"] = selected_path
    paths["multi_anchor_summary_json"] = summary_path
    if result.final_initial_estimates is not None:
        final_anchor_path = output / "mmuad_multi_anchor_final_initial_estimates.csv"
        result.final_initial_estimates.to_csv(final_anchor_path, index=False)
        paths["multi_anchor_final_initial_estimates_csv"] = final_anchor_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk",
        description="condition adaptive MMUAD group selection on multiple trajectories",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument(
        "--initial-estimates",
        action="append",
        required=True,
        help="trajectory hypothesis as NAME=path; repeat for multiple anchors",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument(
        "--anchor-aggregation", choices=ANCHOR_AGGREGATION_CHOICES, default="softmin"
    )
    parser.add_argument("--anchor-softmin-temperature", type=float, default=0.5)
    parser.add_argument("--minimum-anchor-support", type=int, default=1)
    parser.add_argument(
        "--insufficient-anchor-policy",
        choices=INSUFFICIENT_SUPPORT_POLICIES,
        default="neutral",
    )
    parser.add_argument("--final-anchor-policy", choices=FINAL_ANCHOR_POLICIES, default="none")
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
    parser.add_argument("--anchor-selection-weight", type=float, default=1.0)
    parser.add_argument("--anchor-scale-m", type=float, default=10.0)
    parser.add_argument("--anchor-huber-delta", type=float, default=1.0)
    parser.add_argument("--anchor-cost-cap", type=float, default=4.0)
    parser.add_argument("--anchor-time-tolerance-s", type=float, default=0.5)
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
    anchor_config = AnchorConditioningConfig(
        anchor_selection_weight=args.anchor_selection_weight,
        anchor_scale_m=args.anchor_scale_m,
        anchor_huber_delta=args.anchor_huber_delta,
        anchor_cost_cap=args.anchor_cost_cap,
        anchor_time_tolerance_s=args.anchor_time_tolerance_s,
        missing_anchor_policy="neutral",
    )
    multi_anchor_config = MultiAnchorConditioningConfig(
        aggregation=args.anchor_aggregation,
        softmin_temperature=args.anchor_softmin_temperature,
        minimum_anchor_support=args.minimum_anchor_support,
        insufficient_support_policy=args.insufficient_anchor_policy,
        final_anchor_policy=args.final_anchor_policy,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    anchors = _load_anchor_specs(args.initial_estimates)
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_multi_anchor_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        initial_estimates_by_name=anchors,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        anchor_config=anchor_config,
        multi_anchor_config=multi_anchor_config,
        truth=truth,
    )
    paths = write_multi_anchor_posterior_mass_group_topk_outputs(result, args.output_dir)
    print("mmuad_multi_anchor_posterior_mass_group_topk=ok")
    print(f"anchor_count={len(anchors)}")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled", {}
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _aggregate_anchor_costs(
    costs: np.ndarray,
    *,
    aggregation: str,
    temperature: float,
) -> np.ndarray:
    support = np.isfinite(costs)
    counts = support.sum(axis=1)
    out = np.zeros(costs.shape[0], dtype=float)
    valid = counts > 0
    if aggregation == "min":
        out[valid] = np.nanmin(costs[valid], axis=1)
    elif aggregation == "mean":
        out[valid] = np.nanmean(costs[valid], axis=1)
    elif aggregation == "softmin":
        tau = float(temperature)
        values = np.where(support[valid], np.exp(-costs[valid] / tau), 0.0)
        mean_exp = values.sum(axis=1) / counts[valid]
        out[valid] = -tau * np.log(np.maximum(mean_exp, 1.0e-300))
    else:  # pragma: no cover - validated by caller
        raise ValueError(f"unsupported anchor aggregation: {aggregation}")
    return out


def _nan_stat(costs: np.ndarray, function: Any) -> np.ndarray:
    out = np.full(costs.shape[0], np.nan, dtype=float)
    valid = np.isfinite(costs).any(axis=1)
    if valid.any():
        with np.errstate(all="ignore"):
            out[valid] = function(costs[valid], axis=1)
    return out


def _best_anchor_names(costs: np.ndarray, anchor_names: list[str]) -> list[str]:
    result: list[str] = []
    for row in costs:
        finite = np.isfinite(row)
        if not finite.any():
            result.append("")
            continue
        masked = np.where(finite, row, np.inf)
        result.append(str(anchor_names[int(np.argmin(masked))]))
    return result


def _build_final_initial_estimates(
    anchor_coordinate_frames: list[pd.DataFrame],
    normalized_anchors: Mapping[str, pd.DataFrame],
    *,
    policy: str,
) -> pd.DataFrame | None:
    if policy == "none":
        return None
    if policy == "first":
        first = next(iter(normalized_anchors.values()))
        return first.copy().reset_index(drop=True)
    if not anchor_coordinate_frames:
        return None
    rows = pd.concat(anchor_coordinate_frames, ignore_index=True).rename(
        columns={
            "mixture_anchor_x_m": "state_x_m",
            "mixture_anchor_y_m": "state_y_m",
            "mixture_anchor_z_m": "state_z_m",
        }
    )
    return (
        rows.groupby(["sequence_id", "time_s"], as_index=False, sort=True)[
            ["state_x_m", "state_y_m", "state_z_m"]
        ]
        .median()
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )


def _validate_anchor_mapping(
    initial_estimates_by_name: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    anchors: dict[str, pd.DataFrame] = {}
    for raw_name, rows in initial_estimates_by_name.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("trajectory anchor names must be non-empty")
        if name in anchors:
            raise ValueError(f"duplicate trajectory anchor name: {name}")
        anchors[name] = pd.DataFrame(rows).copy()
    if not anchors:
        raise ValueError("at least one trajectory anchor is required")
    return anchors


def _assert_candidate_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    anchor_name: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"candidate row count changed while scoring anchor {anchor_name!r}")
    columns = ["sequence_id", "time_s", "x_m", "y_m", "z_m"]
    left = reference[columns].reset_index(drop=True)
    right = candidate[columns].reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(f"candidate row order changed while scoring anchor {anchor_name!r}")


def _validate_multi_anchor_config(config: MultiAnchorConditioningConfig) -> None:
    if config.aggregation not in ANCHOR_AGGREGATION_CHOICES:
        raise ValueError(f"unsupported anchor aggregation: {config.aggregation}")
    if not np.isfinite(config.softmin_temperature) or config.softmin_temperature <= 0.0:
        raise ValueError("softmin_temperature must be finite and positive")
    if int(config.minimum_anchor_support) < 1:
        raise ValueError("minimum_anchor_support must be at least one")
    if config.insufficient_support_policy not in INSUFFICIENT_SUPPORT_POLICIES:
        raise ValueError(
            "unsupported insufficient anchor support policy: "
            f"{config.insufficient_support_policy}"
        )
    if config.final_anchor_policy not in FINAL_ANCHOR_POLICIES:
        raise ValueError(f"unsupported final anchor policy: {config.final_anchor_policy}")


def _multi_anchor_summary(
    rows: pd.DataFrame,
    *,
    anchor_names: list[str],
    safe_names: Mapping[str, str],
    anchor_config: AnchorConditioningConfig,
    multi_anchor_config: MultiAnchorConditioningConfig,
    final_initial_estimates: pd.DataFrame | None,
) -> dict[str, Any]:
    support = pd.to_numeric(rows["mixture_multi_anchor_support_count"], errors="coerce")
    disagreement = pd.to_numeric(rows["mixture_multi_anchor_cost_std"], errors="coerce")
    return _jsonable(
        {
            "schema": "raft-uav-mmuad-multi-anchor-conditioning-v1",
            "anchor_names": anchor_names,
            "anchor_safe_names": dict(safe_names),
            "anchor_count": int(len(anchor_names)),
            "candidate_rows": int(len(rows)),
            "anchor_config": asdict(anchor_config),
            "multi_anchor_config": asdict(multi_anchor_config),
            "support_count_mean": _finite_mean(support),
            "support_count_min": _finite_min(support),
            "support_count_max": _finite_max(support),
            "cost_std_mean": _finite_mean(disagreement),
            "cost_std_p95": _finite_quantile(disagreement, 0.95),
            "insufficient_support_rows": int(
                (~rows["mixture_multi_anchor_sufficient_support"].astype(bool)).sum()
            ),
            "final_initial_estimate_rows": (
                0 if final_initial_estimates is None else int(len(final_initial_estimates))
            ),
            "truth_used_for_selection": False,
        }
    )


def _load_anchor_specs(specs: list[str]) -> dict[str, pd.DataFrame]:
    anchors: dict[str, pd.DataFrame] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--initial-estimates must use NAME=path syntax")
        name, path_text = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--initial-estimates anchor name must be non-empty")
        if name in anchors:
            raise ValueError(f"duplicate --initial-estimates anchor name: {name}")
        anchors[name] = read_estimate_csv(Path(path_text))
    return anchors


def _unique_safe_name(value: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_") or "anchor"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _finite_values(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return array[np.isfinite(array)]


def _finite_mean(values: pd.Series) -> float:
    array = _finite_values(values)
    return float(array.mean()) if array.size else float("nan")


def _finite_min(values: pd.Series) -> float:
    array = _finite_values(values)
    return float(array.min()) if array.size else float("nan")


def _finite_max(values: pd.Series) -> float:
    array = _finite_values(values)
    return float(array.max()) if array.size else float("nan")


def _finite_quantile(values: pd.Series, quantile: float) -> float:
    array = _finite_values(values)
    return float(np.quantile(array, quantile)) if array.size else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
